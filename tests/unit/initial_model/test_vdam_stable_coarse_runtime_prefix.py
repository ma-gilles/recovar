"""GPU contract for stable-capacity coarse scoring over a logical prefix."""

from itertools import permutations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from recovar.em.dense_single_volume.helpers.fourier_window import (
    make_fourier_window_indices_np,
)
from recovar.em.dense_single_volume.helpers.oversampling import (
    relion_cuda_f32_coarse_posterior,
)
from recovar.em.dense_single_volume.helpers.significance import (
    _plan_coarse_gaussian_square_layout,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def _fma32(left, right, addend):
    return np.asarray(
        np.asarray(left, dtype=np.float64) * np.asarray(right, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _assert_in_exact_atomic_envelope(
    actual,
    reference,
    shifted,
    weight,
    initial_diff2,
    full_to_compact,
):
    """Require every score to be in the complete default-geometry atomic set.

    For 29 translations, RELION's 128-thread block gives every score exactly
    four nonzero lane partials.  Each lane issues one ``atomicAdd`` to that
    score and every remaining thread contributes exact zero.  CUDA serializes
    the four active additions, so all and only the ``4!`` lane permutations
    below are legal realizations.  This is an exhaustive arithmetic oracle,
    not a tolerance or an empirical repeat envelope.
    """

    actual = np.asarray(actual).reshape(
        shifted.shape[0],
        reference.shape[0],
        shifted.shape[1],
    )
    assert shifted.shape[0] == 1
    translation_count = shifted.shape[1]
    assert translation_count == 29
    active_lanes = 128 // translation_count
    assert active_lanes == 4
    lane_sums = np.zeros(
        (active_lanes, reference.shape[0], translation_count),
        dtype=np.float32,
    )
    for chunk_start in range(0, full_to_compact.size, 32):
        for lane in range(active_lanes):
            for pixel_in_chunk in range(lane, 32, active_lanes):
                full_pixel = chunk_start + pixel_in_chunk
                if full_pixel >= full_to_compact.size:
                    break
                compact_pixel = int(full_to_compact[full_pixel])
                if compact_pixel < 0:
                    continue
                diff_real = np.subtract(
                    reference[:, compact_pixel].real[:, None],
                    shifted[0, :, compact_pixel].real[None, :],
                    dtype=np.float32,
                )
                diff_imag = np.subtract(
                    reference[:, compact_pixel].imag[:, None],
                    shifted[0, :, compact_pixel].imag[None, :],
                    dtype=np.float32,
                )
                imag_square = np.multiply(diff_imag, diff_imag, dtype=np.float32)
                square_sum = _fma32(diff_real, diff_real, imag_square)
                half_square_sum = np.multiply(
                    square_sum,
                    np.float32(0.5),
                    dtype=np.float32,
                )
                lane_sums[lane] = _fma32(
                    half_square_sum,
                    weight[0, compact_pixel],
                    lane_sums[lane],
                )

    possible_bits = []
    for order in permutations(range(active_lanes)):
        candidate = np.full(
            (reference.shape[0], translation_count),
            initial_diff2[0],
            dtype=np.float32,
        )
        for lane in order:
            candidate = np.add(candidate, lane_sums[lane], dtype=np.float32)
        possible_bits.append(candidate.view(np.uint32))
    legal = np.any(
        np.stack(possible_bits, axis=0) == actual[0].view(np.uint32)[None],
        axis=0,
    )
    assert np.all(legal), f"{np.count_nonzero(~legal)} scores left atomic envelope"


def _posterior_on_host(scores, gpu_device):
    with jax.default_device(gpu_device):
        result = relion_cuda_f32_coarse_posterior(
            -jnp.asarray(scores).reshape(scores.shape[0], -1),
            adaptive_fraction=0.999,
            max_significants=500,
        )
    return tuple(np.asarray(value) for value in result)


def _make_operands(seed, physical_count):
    rng = np.random.default_rng(seed)
    reference = (
        rng.normal(0.0, 0.02, (16, physical_count))
        + 1j * rng.normal(0.0, 0.02, (16, physical_count))
    ).astype(np.complex64)
    shifted = (
        rng.normal(0.0, 0.02, (1, 29, physical_count))
        + 1j * rng.normal(0.0, 0.02, (1, 29, physical_count))
    ).astype(np.complex64)
    weight = rng.uniform(0.0, 150_000.0, (1, physical_count)).astype(np.float32)
    initial_diff2 = rng.uniform(10_000.0, 20_000.0, 1).astype(np.float32)
    return reference, shifted, weight, initial_diff2


def _layouts():
    image_shape = (128, 128)
    logical_size = 86
    logical_indices, logical_count = make_fourier_window_indices_np(
        image_shape,
        logical_size,
        square=True,
        include_dc=True,
    )
    logical_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=False,
    )
    stable_layout = _plan_coarse_gaussian_square_layout(
        image_shape,
        logical_size,
        logical_indices,
        stable_fourier_window_shapes=True,
    )
    assert stable_layout.physical_square_count > logical_count
    return logical_count, logical_layout, stable_layout


def _configure_cuda(monkeypatch, custom_cuda_lib):
    from recovar import cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setenv("RECOVAR_RELION_VDAM_STABLE_FOURIER_WINDOW_QUANTUM", "32")
    cuda_backproject._cuda_ok = None
    return cuda_backproject


def test_stable_coarse_runtime_prefix_preserves_posterior_support_wordwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Padded execution stays in the exact atomic set with exact decisions."""

    cuda_backproject = _configure_cuda(monkeypatch, custom_cuda_lib)
    logical_count, logical_layout, stable_layout = _layouts()
    reference, shifted, weight, initial_diff2 = _make_operands(
        20260903,
        stable_layout.physical_square_count,
    )

    with jax.default_device(gpu_device):
        direct = cuda_backproject.relion_coarse_diff2_rectangular_f32(
            jnp.asarray(reference[:, :logical_count]),
            jnp.asarray(shifted[:, :, :logical_count]),
            jnp.asarray(weight[:, :logical_count]),
            jnp.asarray(initial_diff2),
            jnp.asarray(logical_layout.full_to_compact_np),
        )
        direct = np.asarray(direct.block_until_ready())
        stable = cuda_backproject.relion_coarse_diff2_rectangular_runtime_f32(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(stable_layout.full_to_compact_np),
            jnp.asarray(logical_count, dtype=jnp.int32),
        )
        stable = np.asarray(stable.block_until_ready())

    compact_operands = (
        reference[:, :logical_count],
        shifted[:, :, :logical_count],
        weight[:, :logical_count],
        initial_diff2,
        logical_layout.full_to_compact_np,
    )
    _assert_in_exact_atomic_envelope(direct, *compact_operands)
    _assert_in_exact_atomic_envelope(stable, *compact_operands)
    np.testing.assert_array_equal(
        np.argmin(stable.reshape(stable.shape[0], -1), axis=1),
        np.argmin(direct.reshape(direct.shape[0], -1), axis=1),
    )
    for stable_value, direct_value in zip(
        _posterior_on_host(stable, gpu_device),
        _posterior_on_host(direct, gpu_device),
    ):
        np.testing.assert_array_equal(stable_value, direct_value)


def test_stable_coarse_runtime_prefix_source16_stays_in_atomic_envelope(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    """Selected source-16 rescoring obeys the same logical-prefix contract."""

    cuda_backproject = _configure_cuda(monkeypatch, custom_cuda_lib)
    logical_count, logical_layout, stable_layout = _layouts()
    reference, shifted, weight, initial_diff2 = _make_operands(
        20260904,
        stable_layout.physical_square_count,
    )
    block_ids = np.zeros((1, 1), dtype=np.int32)

    with jax.default_device(gpu_device):
        direct = cuda_backproject.relion_coarse_diff2_rotation_blocks_f32(
            jnp.asarray(reference[:, :logical_count]),
            jnp.asarray(shifted[:, :, :logical_count]),
            jnp.asarray(weight[:, :logical_count]),
            jnp.asarray(initial_diff2),
            jnp.asarray(block_ids),
            jnp.asarray(logical_layout.full_to_compact_np),
        )
        direct = np.asarray(direct.block_until_ready())
        stable = cuda_backproject.relion_coarse_diff2_rotation_blocks_runtime_f32(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial_diff2),
            jnp.asarray(block_ids),
            jnp.asarray(stable_layout.full_to_compact_np),
            jnp.asarray(logical_count, dtype=jnp.int32),
        )
        stable = np.asarray(stable.block_until_ready())

    compact_operands = (
        reference[:, :logical_count],
        shifted[:, :, :logical_count],
        weight[:, :logical_count],
        initial_diff2,
        logical_layout.full_to_compact_np,
    )
    _assert_in_exact_atomic_envelope(direct, *compact_operands)
    _assert_in_exact_atomic_envelope(stable, *compact_operands)
