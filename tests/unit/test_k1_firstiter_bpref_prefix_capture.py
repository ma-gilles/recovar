from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import recovar.cuda_backproject as cuda_backproject

from recovar.em.sparse_pass2 import firstiter_bpref as sparse
from recovar.em.diagnostics import bpref_diagnostics


def test_deferred_firstiter_bpref_gate_uses_texture_peak_and_force_override(
    monkeypatch,
):
    box800_projector_bytes, box800_accumulator_bytes, box800_peak_bytes = (
        sparse._relion_firstiter_bpref_overlap_bytes(
            projector_shape=(1603, 1603, 802),
            projector_dtype=np.complex64,
            recon_volume_size=2_060_826_418,
            recon_y_dtype=np.complex64,
            recon_ctf_dtype=np.float32,
        )
    )
    assert box800_projector_bytes == 16_486_611_344
    assert box800_accumulator_bytes == 24_729_917_016
    assert box800_peak_bytes == 57_703_139_704

    common = dict(
        relion_firstiter_fused_bpref=True,
        use_relion_projector=True,
        projector_device_owned=True,
        projector_shape=(10, 10, 6),
        projector_dtype=np.complex64,
        recon_volume_size=100,
        recon_y_dtype=np.complex64,
        recon_ctf_dtype=np.float32,
    )
    projector_bytes, accumulator_bytes, peak_bytes = (
        sparse._relion_firstiter_bpref_overlap_bytes(
            projector_shape=common["projector_shape"],
            projector_dtype=common["projector_dtype"],
            recon_volume_size=common["recon_volume_size"],
            recon_y_dtype=common["recon_y_dtype"],
            recon_ctf_dtype=common["recon_ctf_dtype"],
        )
    )
    assert projector_bytes == 4_800
    assert accumulator_bytes == 1_200
    assert peak_bytes == 10_800

    monkeypatch.delenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, raising=False)
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=20_000,
    )
    assert not sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=22_000,
    )
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=22_000,
        allocator_free_memory_bytes=1_000,
    )

    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, "1")
    assert sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=1_000_000,
    )
    with pytest.raises(ValueError, match="host-owned RELION projector"):
        sparse._relion_firstiter_deferred_bpref_enabled(
            **(common | {"projector_device_owned": False}),
            device_memory_bytes=20_000,
        )
    with pytest.raises(ValueError, match="host-owned RELION projector"):
        sparse._relion_firstiter_deferred_bpref_enabled(
            **common,
            device_memory_bytes=20_000,
            diagnostics_active=True,
        )

    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, "0")
    assert not sparse._relion_firstiter_deferred_bpref_enabled(
        **common,
        device_memory_bytes=20_000,
    )


def test_deferred_firstiter_bpref_host_estimate_and_cap(monkeypatch):
    buckets = [
        {"bucket_size": 5, "image_indices": np.asarray([0, 1, 2])},
        {"bucket_size": 7, "image_indices": np.asarray([3])},
    ]
    expected = (
        4 * 11 * (np.dtype(np.complex64).itemsize + np.dtype(np.float32).itemsize)
        + 2 * 11 * np.dtype(np.float32).itemsize
        + (3 * 5 + 1 * 7)
        * (2 * np.dtype(np.float32).itemsize + 9 * np.dtype(np.float32).itemsize)
        + 4 * 3 * np.dtype(np.int64).itemsize
    )
    assert sparse._deferred_firstiter_bpref_estimated_host_bytes(
        buckets,
        n_half=11,
        n_fine_trans=2,
    ) == int(expected)

    # EMPIAR-10202 set 6 half 1: 15,258 particles, box 800, 16 padded
    # rotations, 84 fine translations, and 2,180 seven-particle-or-smaller
    # launch groups.  The halves run serially, so this is the peak retained
    # payload admitted by the production smoke rather than both halves added.
    set6_half1_buckets = [
        {
            "bucket_size": 16,
            "image_indices": np.arange(start, min(start + 7, 15_258)),
        }
        for start in range(0, 15_258, 7)
    ]
    assert sparse._deferred_firstiter_bpref_estimated_host_bytes(
        set6_half1_buckets,
        n_half=320_800,
        n_fine_trans=84,
    ) == 61_625_754_608

    monkeypatch.delenv(
        sparse._RELION_FIRSTITER_DEFERRED_BPREF_MAX_HOST_BYTES_ENV,
        raising=False,
    )
    assert (
        sparse._deferred_firstiter_bpref_max_host_bytes()
        == sparse._DEFAULT_DEFERRED_FIRSTITER_BPREF_MAX_HOST_BYTES
    )
    monkeypatch.setenv(
        sparse._RELION_FIRSTITER_DEFERRED_BPREF_MAX_HOST_BYTES_ENV,
        "12345",
    )
    assert sparse._deferred_firstiter_bpref_max_host_bytes() == 12345


def test_scoped_bpref_configured_target_is_not_an_active_diagnostic():
    flags = {
        "device_signature_configured": True,
        "sequential_translation_reduction": False,
        "per_particle_launches": False,
        "fused_atomics": False,
        "high_precision_operand_bundle": False,
    }
    assert not bpref_diagnostics._scoped_bpref_diagnostics_active(flags)
    flags["fused_atomics"] = True
    assert bpref_diagnostics._scoped_bpref_diagnostics_active(flags)


def test_release_deferred_firstiter_projection_buffers_deletes_unique_values_once(
    caplog,
):
    events = []

    class Buffer:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        def delete(self):
            events.append(self.name)
            if self.fail:
                raise RuntimeError("already released")

    projector = Buffer("projector")
    cached = Buffer("cache")
    already_released = Buffer("stale", fail=True)
    sparse._release_deferred_firstiter_projection_buffers(
        projector,
        {
            "score": cached,
            "recon": cached,
            "recon_abs2": already_released,
        },
    )
    assert events == ["projector", "cache", "stale"]
    assert "could not explicitly release Buffer: already released" in caplog.text


def test_deferred_firstiter_bpref_snapshot_and_replay_preserve_bits_and_order(
    monkeypatch,
):
    first_images = jnp.asarray([[1 + 2j, 3 + 4j]], dtype=jnp.complex64)
    first_ctf = jnp.asarray([[5.0, 6.0]], dtype=jnp.float32)
    first_noise = jnp.asarray([7.0, 8.0], dtype=jnp.float32)
    first_posterior = jnp.asarray([[[0.0, 1.0]]], dtype=jnp.float32)
    first_rotations = jnp.asarray(np.eye(3, dtype=np.float32)[None, None])
    second_images = jnp.asarray([[9 + 10j, 11 + 12j]], dtype=jnp.complex64)
    second_ctf = jnp.asarray([[13.0, 14.0]], dtype=jnp.float32)
    second_noise = jnp.asarray([15.0, 16.0], dtype=jnp.float32)
    second_posterior = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    second_rotations = jnp.asarray((-np.eye(3, dtype=np.float32))[None, None])

    first = sparse._stage_deferred_firstiter_bpref_batch(
        raw_images=first_images,
        raw_ctf=first_ctf,
        raw_minvsigma2=first_noise,
        posterior=first_posterior,
        rotations=first_rotations,
        actual_counts=np.asarray([1]),
        particle_half_local_indices=np.asarray([0]),
        particle_original_indices=np.asarray([10]),
    )
    second = sparse._stage_deferred_firstiter_bpref_batch(
        raw_images=second_images,
        raw_ctf=second_ctf,
        raw_minvsigma2=second_noise,
        posterior=second_posterior,
        rotations=second_rotations,
        actual_counts=np.asarray([1]),
        particle_half_local_indices=np.asarray([1]),
        particle_original_indices=np.asarray([20]),
    )
    np.testing.assert_array_equal(first.raw_images, np.asarray(first_images))
    np.testing.assert_array_equal(first.raw_ctf, np.asarray(first_ctf))
    np.testing.assert_array_equal(first.raw_minvsigma2, np.asarray(first_noise))
    np.testing.assert_array_equal(first.posterior, np.asarray(first_posterior))
    np.testing.assert_array_equal(first.rotations, np.asarray(first_rotations))
    assert sparse._deferred_firstiter_bpref_batch_nbytes(first) == sum(
        value.nbytes for value in first
    )

    calls = []
    replay_boundaries = []

    def record_replay_boundary(values):
        replay_boundaries.append(tuple(np.asarray(value).copy() for value in values))
        return values

    monkeypatch.setattr(sparse.jax, "block_until_ready", record_replay_boundary)

    def fake_accumulate_split(
        raw_images,
        raw_ctf,
        raw_minvsigma2,
        posterior,
        rotations,
        actual_counts,
        particle_half_local_indices,
        particle_original_indices,
        data_volume_real,
        data_volume_imag,
        weight_volume,
        **kwargs,
    ):
        calls.append(
            {
                "raw_images": np.asarray(raw_images).copy(),
                "raw_ctf": np.asarray(raw_ctf).copy(),
                "raw_minvsigma2": np.asarray(raw_minvsigma2).copy(),
                "posterior": np.asarray(posterior).copy(),
                "rotations": np.asarray(rotations).copy(),
                "actual_counts": np.asarray(actual_counts).copy(),
                "particle_half_local_indices": np.asarray(
                    particle_half_local_indices,
                ).copy(),
                "particle_original_indices": np.asarray(
                    particle_original_indices,
                ).copy(),
                "kwargs": kwargs,
            }
        )
        ordinal = np.float32(np.asarray(particle_original_indices)[0])
        return (
            data_volume_real * np.float32(10.0) + ordinal,
            data_volume_imag * np.float32(10.0) - ordinal,
            weight_volume * np.float32(10.0) + ordinal,
        )

    monkeypatch.setattr(
        sparse,
        "_accumulate_relion_firstiter_bpref_fused_split",
        fake_accumulate_split,
    )
    data_real, data_imag, weight = sparse._replay_deferred_firstiter_bpref_batches(
        [first, second],
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        jnp.zeros(1, dtype=jnp.float32),
        centered_pixel_indices=np.arange(2, dtype=np.int32),
        fftw_pixel_indices=np.arange(2, dtype=np.int32),
        translation_angles=np.zeros((2, 2), dtype=np.float32),
        physical_image_shape=(2, 2),
        volume_shape=(1,),
        max_r=1.0,
        adaptive_fraction=0.999,
    )

    np.testing.assert_array_equal(np.asarray(data_real), np.asarray([120], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(data_imag), np.asarray([-120], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(weight), np.asarray([120], dtype=np.float32))
    assert [call["particle_original_indices"].item() for call in calls] == [10, 20]
    assert len(replay_boundaries) == 2
    np.testing.assert_array_equal(
        replay_boundaries[0][0],
        np.asarray([10], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        replay_boundaries[0][1],
        np.asarray([-10], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        replay_boundaries[1][0],
        np.asarray([120], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        replay_boundaries[1][1],
        np.asarray([-120], dtype=np.float32),
    )
    for expected, actual in zip((first, second), calls, strict=True):
        for field in sparse.DeferredFirstiterBPrefBatch._fields:
            np.testing.assert_array_equal(actual[field], getattr(expected, field))


def test_split_firstiter_bpref_normalization_is_donated_and_bitwise_exact():
    assert (
        sparse._normalize_split_relion_firstiter_bpref_accumulators._jit_info.donate_argnums
        == (0, 1, 2)
    )
    component_pairs = np.asarray(
        [
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, 0.0),
            (-0.0, -0.0),
            (0.0, 2.0),
            (0.0, -2.0),
            (-0.0, 2.0),
            (-0.0, -2.0),
            (2.0, 0.0),
            (2.0, -0.0),
            (-2.0, 0.0),
            (-2.0, -0.0),
            (1.25, -2.5),
            (-(2**24), 3.0),
        ],
        dtype=np.float32,
    )
    data = np.empty(component_pairs.shape[0], dtype=np.complex64)
    data.real = component_pairs[:, 0]
    data.imag = component_pairs[:, 1]
    weight = np.asarray([0.0, -0.0, 3.5, 2**24], dtype=np.float32)
    weight = np.resize(weight, data.shape).astype(np.float32, copy=False)
    fft_size = np.float32(640000.0)
    expected_data = np.asarray(-jnp.asarray(data) / fft_size)
    expected_weight = np.asarray(
        jnp.asarray(weight) / np.float32(fft_size * fft_size),
    )

    real_out, imag_out, weight_out = (
        sparse._normalize_split_relion_firstiter_bpref_accumulators(
            jnp.asarray(data.real.copy()),
            jnp.asarray(data.imag.copy()),
            jnp.asarray(weight.copy()),
            fft_size,
            np.float32(fft_size * fft_size),
        )
    )
    data_out = np.empty(data.shape, dtype=np.complex64)
    data_out.real = np.asarray(real_out)
    data_out.imag = np.asarray(imag_out)
    np.testing.assert_array_equal(data_out.view(np.uint32), expected_data.view(np.uint32))
    np.testing.assert_array_equal(
        np.asarray(weight_out).view(np.uint32),
        expected_weight.view(np.uint32),
    )


@pytest.mark.gpu
def test_split_firstiter_bpref_normalization_aliases_all_gpu_inputs(gpu_device):
    data = np.empty(8, dtype=np.complex64)
    data.real = np.asarray([0.0, 0.0, -0.0, -0.0, 2.0, 2.0, -2.0, -2.0])
    data.imag = np.asarray([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0])
    weight = np.asarray([0.0, -0.0, 1.0, -1.0, 3.5, 2**24, 0.25, 17.0], dtype=np.float32)
    fft_size = np.float32(640000.0)
    expected_data = np.asarray(
        jax.device_get(-jax.device_put(data, gpu_device) / fft_size),
    )
    expected_weight = np.asarray(
        jax.device_get(
            jax.device_put(weight, gpu_device) / np.float32(fft_size * fft_size),
        ),
    )

    real_input = jax.device_put(data.real.copy(), gpu_device)
    imag_input = jax.device_put(data.imag.copy(), gpu_device)
    weight_input = jax.device_put(weight.copy(), gpu_device)
    input_pointers = {
        real_input.unsafe_buffer_pointer(),
        imag_input.unsafe_buffer_pointer(),
        weight_input.unsafe_buffer_pointer(),
    }
    real_out, imag_out, weight_out = (
        sparse._normalize_split_relion_firstiter_bpref_accumulators(
            real_input,
            imag_input,
            weight_input,
            fft_size,
            np.float32(fft_size * fft_size),
        )
    )
    jax.block_until_ready((real_out, imag_out, weight_out))
    output_pointers = {
        real_out.unsafe_buffer_pointer(),
        imag_out.unsafe_buffer_pointer(),
        weight_out.unsafe_buffer_pointer(),
    }

    assert len(input_pointers) == 3
    assert output_pointers == input_pointers
    data_out = np.empty(data.shape, dtype=np.complex64)
    data_out.real = np.asarray(real_out)
    data_out.imag = np.asarray(imag_out)
    np.testing.assert_array_equal(
        data_out.view(np.uint32),
        expected_data.view(np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(weight_out).view(np.uint32),
        expected_weight.view(np.uint32),
    )


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
        bpref_diagnostics,
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
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 1)
    captures = []
    monkeypatch.setattr(
        bpref_diagnostics,
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


def test_firstiter_fused_bpref_defaults_only_inside_complete_fresh_k1_guard(monkeypatch):
    monkeypatch.delenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", raising=False)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 2)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )

    assert sparse._relion_firstiter_fused_bpref_enabled(**kwargs)
    assert not sparse._relion_firstiter_fused_bpref_enabled(
        **{**kwargs, "winner_take_all": False}
    )


def test_firstiter_fused_bpref_override_can_disable_but_not_expand_scope(monkeypatch):
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "iteration", 1)
    monkeypatch.setitem(bpref_diagnostics._bpref_contribution_context, "half", 1)
    kwargs = dict(
        fresh_k1_guard=True,
        winner_take_all=True,
        preserve_bpref_particle_order=True,
        relion_exact_bpref_operands=True,
        use_relion_x_half_mstep=True,
        score_only=False,
    )
    monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "0")
    assert not sparse._relion_firstiter_fused_bpref_enabled(**kwargs)

    monkeypatch.setenv("RECOVAR_K1_RELION_FIRSTITER_FUSED_BPREF", "1")
    with pytest.raises(ValueError, match="requires the fresh K=1"):
        sparse._relion_firstiter_fused_bpref_enabled(
            **{**kwargs, "fresh_k1_guard": False}
        )



@pytest.mark.gpu
def test_deferred_firstiter_bpref_replay_matches_eager_native_accumulators_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    cuda_backproject._cuda_ok = None

    image_shape = (4, 4)
    volume_shape = (7, 7, 7)
    volume_size = 7 * 7 * 4
    centered_pixel_indices = np.arange(12, dtype=np.int32)
    fftw_pixel_indices = np.arange(12, dtype=np.int32)
    translation_angles = np.zeros((1, 2), dtype=np.float32)
    rotations = np.eye(3, dtype=np.float32)[None, None]

    def launch_group(tag, original_index):
        image = np.zeros((1, 12), dtype=np.complex64)
        image[0, 0] = np.complex64(tag)
        ctf = np.zeros((1, 12), dtype=np.float32)
        ctf[0, 0] = np.float32(2.0)
        return dict(
            raw_images=image,
            raw_ctf=ctf,
            raw_minvsigma2=np.full(12, np.float32(0.5), dtype=np.float32),
            posterior=np.ones((1, 1, 1), dtype=np.float32),
            rotations=rotations,
            actual_counts=np.asarray([1], dtype=np.int64),
            particle_half_local_indices=np.asarray(
                [original_index],
                dtype=np.int64,
            ),
            particle_original_indices=np.asarray(
                [original_index],
                dtype=np.int64,
            ),
        )

    groups = [
        launch_group(np.complex64(2**24 - (2**24) * 1j), 0),
        launch_group(np.complex64(1 + 1j), 1),
        launch_group(np.complex64(-(2**24) + (2**24) * 1j), 2),
    ]
    staged = [
        sparse._stage_deferred_firstiter_bpref_batch(**group)
        for group in groups
    ]
    common = dict(
        centered_pixel_indices=centered_pixel_indices,
        fftw_pixel_indices=fftw_pixel_indices,
        translation_angles=translation_angles,
        physical_image_shape=image_shape,
        volume_shape=volume_shape,
        max_r=2.0,
        adaptive_fraction=0.999,
    )

    with cuda_backproject.jax.default_device(gpu_device):
        eager_data = jnp.zeros(volume_size, dtype=jnp.complex64)
        eager_weight = jnp.zeros(volume_size, dtype=jnp.float32)
        for group in groups:
            eager_data, eager_weight = (
                sparse._accumulate_relion_firstiter_bpref_fused(
                    data_volume=eager_data,
                    weight_volume=eager_weight,
                    **group,
                    **common,
                )
            )
        deferred_real, deferred_imag, deferred_weight = (
            sparse._replay_deferred_firstiter_bpref_batches(
                staged,
                jnp.zeros(volume_size, dtype=jnp.float32),
                jnp.zeros(volume_size, dtype=jnp.float32),
                jnp.zeros(volume_size, dtype=jnp.float32),
                **common,
            )
        )
        cuda_backproject.jax.block_until_ready(
            (eager_data, eager_weight, deferred_real, deferred_imag, deferred_weight),
        )

    np.testing.assert_array_equal(
        np.asarray(deferred_real),
        np.asarray(eager_data).real,
    )
    np.testing.assert_array_equal(
        np.asarray(deferred_imag),
        np.asarray(eager_data).imag,
    )
    np.testing.assert_array_equal(
        np.asarray(deferred_weight),
        np.asarray(eager_weight),
    )
    cancellation_offset = 3 * (7 * 4) + 3 * 4
    assert np.asarray(eager_data)[cancellation_offset] == np.complex64(0.0 + 1.0j)
    assert np.asarray(eager_weight)[cancellation_offset] == np.float32(6.0)



@pytest.mark.parametrize('name', [
    'RECOVAR_SPARSE_PASS2_NATIVE_DUMP_DIR',
    'RECOVAR_BPREF_ACCUMULATOR_DELTA_DUMP_DIR',
    'RECOVAR_PASS2_DUMP_DIR',
])
def test_deferred_firstiter_preserves_requested_capture_route(monkeypatch, name):
    monkeypatch.setenv(name, '/requested/capture')
    active = bpref_diagnostics._relion_firstiter_bpref_diagnostics_active(
        bpref_device_signature_active=False,
    )
    assert active
    common = dict(
        relion_firstiter_fused_bpref=True, use_relion_projector=True,
        projector_device_owned=True, projector_shape=(1603, 1603, 802),
        projector_dtype=np.complex64, recon_volume_size=2_060_826_418,
        recon_y_dtype=np.complex64, recon_ctf_dtype=np.float32,
        device_memory_bytes=80 * 1024**3, diagnostics_active=active,
    )
    monkeypatch.delenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, raising=False)
    assert not sparse._relion_firstiter_deferred_bpref_enabled(**common)
    monkeypatch.setenv(sparse._RELION_FIRSTITER_DEFERRED_BPREF_ENV, '1')
    with pytest.raises(ValueError):
        sparse._relion_firstiter_deferred_bpref_enabled(**common)
