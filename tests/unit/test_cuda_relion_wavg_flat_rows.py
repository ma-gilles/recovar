"""Flat-row RELION Wavg kernels against their rectangular twins.

The flat-row kernels replace the rectangular ``[image, rotation]`` block grid
with one packed candidate-row axis whose image address comes from
``row_image_ids``.  Everything else -- the per-pixel binary32 arithmetic, the
translation-storage order and the runtime logical-pixel prefix -- is the
rectangular body, so a flattened rectangular problem must return bitwise
identical results.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.unit


def _complex64(rng, shape):
    return (
        rng.normal(0.0, 0.7, shape) + 1j * rng.normal(0.0, 0.7, shape)
    ).astype(np.complex64)


def _exact_float32(rng, shape, *, quantum=1.0 / 256.0, span=64):
    """Draw float32 values whose partial sums stay exactly representable.

    Every value is an integer multiple of ``2**-8`` bounded by ``span``, so a
    handful of them add without rounding.  Sums are then order-independent and
    a bitwise comparison against the rectangular atomic kernel measures the
    addressing rather than the (hardware-scheduled) atomic issue order.
    """

    integers = rng.integers(-span * 256, span * 256 + 1, size=shape)
    return (integers * quantum).astype(np.float32)


def _rectangular_operands(rng, *, batch_size, rotation_count, translation_count, pixel_capacity):
    return {
        "projections": _complex64(rng, (batch_size, rotation_count, pixel_capacity)),
        "raw_ctf": rng.normal(0.0, 1.5, (batch_size, pixel_capacity)).astype(np.float32),
        "scale": rng.uniform(0.4, 2.5, (batch_size,)).astype(np.float32),
        "shifted_images": _complex64(rng, (batch_size, translation_count, pixel_capacity)),
        "posterior": rng.uniform(0.0, 1.0, (batch_size, rotation_count, translation_count)).astype(
            np.float32
        ),
    }


def _cuda_backproject(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    if not em_cuda_kernels.relion_wavg_sequential_runtime_flat_rows_triplet_f32_supported():
        pytest.skip("loaded CUDA library predates the flat-row Wavg targets")
    if not (
        em_cuda_kernels.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32_supported()
    ):
        pytest.skip("loaded CUDA library predates the flat-row Wavg atomics")
    from recovar.em.cuda import kernels as em_cuda_kernels

    return em_cuda_kernels


def _assert_bitwise(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
    )


@pytest.mark.gpu
@pytest.mark.parametrize("logical_pixel_count", [13, 8])
def test_flat_rows_wavg_matches_flattened_rectangular_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device, logical_pixel_count
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(4801 + logical_pixel_count)
    batch_size, rotation_count, translation_count, pixel_capacity = 3, 4, 5, 13
    operands = _rectangular_operands(
        rng,
        batch_size=batch_size,
        rotation_count=rotation_count,
        translation_count=translation_count,
        pixel_capacity=pixel_capacity,
    )
    row_image_ids = np.repeat(np.arange(batch_size, dtype=np.int32), rotation_count)

    with jax.default_device(gpu_device):
        expected = cuda_backproject.relion_wavg_sequential_runtime_triplet_f32(
            jnp.asarray(operands["projections"]),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(operands["posterior"]),
            jnp.asarray(logical_pixel_count, dtype=jnp.int32),
        )
        actual = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(operands["projections"].reshape(-1, pixel_capacity)),
            jnp.asarray(row_image_ids),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(operands["posterior"].reshape(-1, translation_count)),
            jnp.asarray(logical_pixel_count, dtype=jnp.int32),
        )
        expected, actual = jax.block_until_ready((expected, actual))

    expected = np.asarray(expected).reshape(-1, pixel_capacity, 3)
    _assert_bitwise(actual, expected)
    if logical_pixel_count < pixel_capacity:
        assert np.all(np.asarray(actual)[:, logical_pixel_count:, :] == 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("logical_pixel_count", [11, 6])
def test_flat_rows_wavg_ragged_and_padded_rows(
    monkeypatch, custom_cuda_lib, gpu_device, logical_pixel_count
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(9107 + logical_pixel_count)
    batch_size, rotation_count, translation_count, pixel_capacity = 3, 4, 6, 11
    operands = _rectangular_operands(
        rng,
        batch_size=batch_size,
        rotation_count=rotation_count,
        translation_count=translation_count,
        pixel_capacity=pixel_capacity,
    )
    # Ragged rows per image, with padding in the middle and at the end.
    row_image_ids = np.asarray([0, 0, 0, 1, -1, 2, 2, -1], dtype=np.int32)
    row_rotation_ids = np.asarray([0, 2, 3, 1, 0, 0, 3, 0], dtype=np.int32)
    valid = row_image_ids >= 0

    flat_projections = operands["projections"][row_image_ids, row_rotation_ids]
    flat_posterior = operands["posterior"][row_image_ids, row_rotation_ids]
    # Padding rows must read nothing: give them values that would be visible.
    flat_projections[~valid] = np.complex64(5.0 + 3.0j)
    flat_posterior[~valid] = np.float32(7.0)

    with jax.default_device(gpu_device):
        rectangular = cuda_backproject.relion_wavg_sequential_runtime_triplet_f32(
            jnp.asarray(operands["projections"]),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(operands["posterior"]),
            jnp.asarray(logical_pixel_count, dtype=jnp.int32),
        )
        actual = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(flat_projections),
            jnp.asarray(row_image_ids),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(flat_posterior),
            jnp.asarray(logical_pixel_count, dtype=jnp.int32),
        )
        rectangular, actual = jax.block_until_ready((rectangular, actual))

    rectangular = np.asarray(rectangular)
    actual = np.asarray(actual)
    expected_valid = rectangular[row_image_ids[valid], row_rotation_ids[valid]]
    _assert_bitwise(actual[valid], expected_valid)
    assert np.all(actual[~valid] == 0.0)


@pytest.mark.gpu
def test_flat_rows_wavg_out_of_range_row_id_fails_closed(
    monkeypatch, custom_cuda_lib, gpu_device
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(2213)
    batch_size, rotation_count, translation_count, pixel_capacity = 2, 1, 3, 7
    operands = _rectangular_operands(
        rng,
        batch_size=batch_size,
        rotation_count=rotation_count,
        translation_count=translation_count,
        pixel_capacity=pixel_capacity,
    )
    row_image_ids = np.asarray([0, batch_size, 1], dtype=np.int32)
    flat_projections = operands["projections"][[0, 0, 1], 0]
    flat_posterior = operands["posterior"][[0, 0, 1], 0]

    with jax.default_device(gpu_device):
        actual = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(flat_projections),
            jnp.asarray(row_image_ids),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(flat_posterior),
            jnp.asarray(pixel_capacity, dtype=jnp.int32),
        )
        actual = jax.block_until_ready(actual)

    actual = np.asarray(actual)
    assert np.all(np.isnan(actual[1]))
    assert not np.any(np.isnan(actual[[0, 2]]))


@pytest.mark.gpu
@pytest.mark.parametrize("logical_pixel_count", [7, 4])
def test_flat_rows_wavg_atomics_match_flattened_rectangular_bitwise(
    monkeypatch, custom_cuda_lib, gpu_device, logical_pixel_count
):
    """Exactly representable summands make the accumulation order irrelevant.

    The flat-row launch keeps one block per row with grid.x = row, so a
    flattened rectangular problem issues the same multiset of per-cell atomic
    adds under the same linear block index.  The order those adds land in is
    hardware scheduled, and the rectangular kernel is itself not bitwise
    reproducible at realistic rotation counts, so the asserted comparison uses
    multiples of 2**-8 whose sums are exact in binary32.
    """

    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(3311 + logical_pixel_count)
    batch_size, rotation_count, pixel_capacity = 3, 5, 7
    terms = _exact_float32(rng, (batch_size, rotation_count, pixel_capacity, 3))
    accumulator = _exact_float32(rng, (batch_size, pixel_capacity, 3))
    row_image_ids = np.repeat(np.arange(batch_size, dtype=np.int32), rotation_count)

    with jax.default_device(gpu_device):
        expected = cuda_backproject.relion_wavg_rotation_atomic_runtime_triplet_add_f32(
            jnp.asarray(terms),
            jnp.asarray(accumulator),
            jnp.asarray(logical_pixel_count, dtype=jnp.int32),
        )
        actual = (
            cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
                jnp.asarray(terms.reshape(-1, pixel_capacity, 3)),
                jnp.asarray(row_image_ids),
                jnp.asarray(accumulator),
                jnp.asarray(logical_pixel_count, dtype=jnp.int32),
            )
        )
        expected, actual = jax.block_until_ready((expected, actual))

    _assert_bitwise(actual, expected)
    if logical_pixel_count < pixel_capacity:
        _assert_bitwise(
            np.asarray(actual)[:, logical_pixel_count:, :],
            accumulator[:, logical_pixel_count:, :],
        )


@pytest.mark.gpu
def test_flat_rows_wavg_atomics_ragged_and_padded_rows(
    monkeypatch, custom_cuda_lib, gpu_device
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(6677)
    batch_size, rotation_count, pixel_capacity = 3, 4, 9
    row_image_ids = np.asarray([0, 0, 0, 1, -1, 2, 2, -1], dtype=np.int32)
    row_rotation_ids = np.asarray([0, 1, 2, 0, 0, 0, 1, 0], dtype=np.int32)
    valid = row_image_ids >= 0

    flat_terms = _exact_float32(rng, (row_image_ids.size, pixel_capacity, 3))
    accumulator = _exact_float32(rng, (batch_size, pixel_capacity, 3))
    # The rectangular twin sees the same contributions; unused slots are zero,
    # and padding rows must contribute nothing at all.
    rect_terms = np.zeros((batch_size, rotation_count, pixel_capacity, 3), dtype=np.float32)
    rect_terms[row_image_ids[valid], row_rotation_ids[valid]] = flat_terms[valid]

    with jax.default_device(gpu_device):
        expected = cuda_backproject.relion_wavg_rotation_atomic_runtime_triplet_add_f32(
            jnp.asarray(rect_terms),
            jnp.asarray(accumulator),
            jnp.asarray(pixel_capacity, dtype=jnp.int32),
        )
        actual = (
            cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
                jnp.asarray(flat_terms),
                jnp.asarray(row_image_ids),
                jnp.asarray(accumulator),
                jnp.asarray(pixel_capacity, dtype=jnp.int32),
            )
        )
        expected, actual = jax.block_until_ready((expected, actual))

    _assert_bitwise(actual, expected)


@pytest.mark.gpu
def test_flat_rows_wavg_all_padding_rows_leave_operands_untouched(
    monkeypatch, custom_cuda_lib, gpu_device
):
    cuda_backproject = _cuda_backproject(monkeypatch, custom_cuda_lib)
    rng = np.random.default_rng(1451)
    batch_size, translation_count, pixel_capacity = 2, 3, 5
    operands = _rectangular_operands(
        rng,
        batch_size=batch_size,
        rotation_count=1,
        translation_count=translation_count,
        pixel_capacity=pixel_capacity,
    )
    row_image_ids = np.asarray([-1, -1], dtype=np.int32)
    accumulator = _exact_float32(rng, (batch_size, pixel_capacity, 3))
    terms = _exact_float32(rng, (2, pixel_capacity, 3))

    with jax.default_device(gpu_device):
        triplets = cuda_backproject.relion_wavg_sequential_runtime_flat_rows_triplet_f32(
            jnp.asarray(np.full((2, pixel_capacity), 5.0 + 3.0j, dtype=np.complex64)),
            jnp.asarray(row_image_ids),
            jnp.asarray(operands["raw_ctf"]),
            jnp.asarray(operands["scale"]),
            jnp.asarray(operands["shifted_images"]),
            jnp.asarray(np.full((2, translation_count), 7.0, dtype=np.float32)),
            jnp.asarray(pixel_capacity, dtype=jnp.int32),
        )
        accumulated = (
            cuda_backproject.relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32(
                jnp.asarray(terms),
                jnp.asarray(row_image_ids),
                jnp.asarray(accumulator),
                jnp.asarray(pixel_capacity, dtype=jnp.int32),
            )
        )
        triplets, accumulated = jax.block_until_ready((triplets, accumulated))

    assert np.all(np.asarray(triplets) == 0.0)
    _assert_bitwise(accumulated, accumulator)


def test_flat_rows_wavg_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_wavg_sequential_runtime_flat_rows_triplet_f32.__wrapped__(
            jnp.zeros((1, 1), dtype=jnp.complex64),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 1, 1), dtype=jnp.complex64),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda a: a.__setitem__(0, jnp.zeros((1, 1), dtype=jnp.float32)), "projections"),
        (lambda a: a.__setitem__(1, jnp.zeros((1,), dtype=jnp.float32)), "row_image_ids"),
        (lambda a: a.__setitem__(2, jnp.zeros((1, 2), dtype=jnp.float32)), "raw_ctf"),
        (lambda a: a.__setitem__(3, jnp.ones((2,), dtype=jnp.float32)), "scale"),
        (lambda a: a.__setitem__(4, jnp.zeros((1, 1, 2), dtype=jnp.complex64)), "shifted_images"),
        (lambda a: a.__setitem__(5, jnp.zeros((2, 1), dtype=jnp.float32)), "posterior"),
    ],
)
def test_flat_rows_wavg_rejects_inconsistent_operands(mutate, message):
    from recovar.em.cuda import kernels as em_cuda_kernels

    args = [
        jnp.zeros((1, 1), dtype=jnp.complex64),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.zeros((1, 1, 1), dtype=jnp.complex64),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
    ]
    mutate(args)
    with pytest.raises(ValueError, match=message):
        em_cuda_kernels.relion_wavg_sequential_runtime_flat_rows_triplet_f32.__wrapped__(*args)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda a: a.__setitem__(0, jnp.zeros((1, 1, 2), dtype=jnp.float32)), "terms"),
        (lambda a: a.__setitem__(1, jnp.zeros((2,), dtype=jnp.int32)), "row_image_ids"),
        (lambda a: a.__setitem__(2, jnp.zeros((1, 2, 3), dtype=jnp.float32)), "accumulator"),
    ],
)
def test_flat_rows_wavg_atomics_reject_inconsistent_operands(mutate, message):
    from recovar.em.cuda import kernels as em_cuda_kernels

    args = [
        jnp.zeros((1, 1, 3), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, 1, 3), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
    ]
    mutate(args)
    with pytest.raises(ValueError, match=message):
        wrapped = (
            em_cuda_kernels
            .relion_wavg_rotation_atomic_runtime_flat_rows_triplet_add_f32
            .__wrapped__
        )
        wrapped(*args)
