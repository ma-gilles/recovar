"""Focused tests for RELION's accelerated real-space preprocessing FFI."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _use_custom_cuda_lib(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)


def _zero_fill_shift(images, factors, shifts):
    normalized = images.astype(np.float32) * factors[:, None, None]
    out = np.zeros_like(normalized)
    height, width = images.shape[-2:]
    for row, (dx, dy) in enumerate(shifts.tolist()):
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            continue
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        out[row, dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = normalized[
            row, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def test_relion_cuda_normalize_shift_is_bit_exact(gpu_device):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    rng = np.random.default_rng(20260714)
    images = rng.standard_normal((3, 32, 32)).astype(np.float32)
    factors = np.asarray([0.9988004, 1.0, 1.03125], dtype=np.float32)
    shifts = np.asarray([[-1, -1], [0, 0], [3, -2]], dtype=np.int32)
    expected = _zero_fill_shift(images, factors, shifts)

    with jax.default_device(gpu_device):
        normalized_shifted, unmasked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=False,
        )

    np.testing.assert_array_equal(np.asarray(normalized_shifted), expected)
    np.testing.assert_array_equal(np.asarray(unmasked), expected)


def test_relion_cuda_softmask_has_constant_exterior_and_finite_output(gpu_device):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    rng = np.random.default_rng(17)
    images = rng.standard_normal((2, 64, 64)).astype(np.float32)
    factors = np.asarray([0.97, 1.04], dtype=np.float32)
    shifts = np.asarray([[2, 1], [-3, 0]], dtype=np.int32)

    with jax.default_device(gpu_device):
        normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=20.0,
            cosine_width=5.0,
            apply_mask=True,
        )

    normalized_shifted = np.asarray(normalized_shifted)
    masked = np.asarray(masked)
    np.testing.assert_array_equal(normalized_shifted, _zero_fill_shift(images, factors, shifts))
    assert np.all(np.isfinite(masked))
    yy, xx = np.meshgrid(np.arange(64) - 32, np.arange(64) - 32, indexing="ij")
    exterior = np.sqrt(xx * xx + yy * yy) > 25.0
    for row in range(masked.shape[0]):
        np.testing.assert_array_equal(masked[row][exterior], np.full(np.count_nonzero(exterior), masked[row, 0, 0]))


@pytest.mark.parametrize("native_lane_reduction", [False, True])
def test_relion_cuda_softmask_repeats_bit_exactly(gpu_device, native_lane_reduction):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    rng = np.random.default_rng(20260731)
    images = rng.standard_normal((3, 128, 128)).astype(np.float32)
    factors = np.asarray([0.9988004, 1.0, 1.03125], dtype=np.float32)
    shifts = np.asarray([[-1, -1], [0, 0], [3, -2]], dtype=np.int32)

    masked_repeats = []
    with jax.default_device(gpu_device):
        for _ in range(4):
            _normalized_shifted, masked = relion_preprocess_real_f32(
                jnp.asarray(images),
                jnp.asarray(factors),
                jnp.asarray(shifts),
                radius=23.529411,
                cosine_width=5.0,
                apply_mask=True,
                native_lane_reduction=native_lane_reduction,
            )
            masked_repeats.append(np.asarray(masked))

    for repeat in masked_repeats[1:]:
        np.testing.assert_array_equal(repeat, masked_repeats[0])


@pytest.mark.parametrize("radius,cosine_width", [(1.0e-6, 1.0), (15.999, 1.0e-4)])
def test_relion_cuda_softmask_boundary_radii_remain_finite(radius, cosine_width, gpu_device):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    rng = np.random.default_rng(23)
    images = rng.standard_normal((2, 32, 32)).astype(np.float32)
    with jax.default_device(gpu_device):
        _normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.ones((2,), dtype=jnp.float32),
            jnp.asarray([[1, -1], [-2, 2]], dtype=jnp.int32),
            radius=radius,
            cosine_width=cosine_width,
            apply_mask=True,
        )

    assert np.all(np.isfinite(np.asarray(masked)))


def test_relion_cuda_softmask_fails_closed_with_zero_background_area(gpu_device):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    with jax.default_device(gpu_device), pytest.raises(jax.errors.JaxRuntimeError, match="CUDA: invalid argument"):
        _normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.ones((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=100.0,
            cosine_width=1.0,
            apply_mask=True,
        )
        masked.block_until_ready()


@pytest.mark.parametrize(
    "images,factors,shifts,error",
    [
        (np.zeros((1, 8, 8), dtype=np.float64), np.ones(1, dtype=np.float32), np.zeros((1, 2), dtype=np.int32), "images must be float32"),
        (np.zeros((1, 8, 8), dtype=np.float32), np.ones(1, dtype=np.float64), np.zeros((1, 2), dtype=np.int32), "normalization_factors must be float32"),
        (np.zeros((1, 8, 8), dtype=np.float32), np.ones(1, dtype=np.float32), np.zeros((1, 2), dtype=np.int64), "integer_shifts must be int32"),
    ],
)
def test_relion_cuda_preprocess_rejects_wrong_dtypes(images, factors, shifts, error, gpu_device):
    from recovar.cuda_backproject import relion_preprocess_real_f32

    with jax.default_device(gpu_device), pytest.raises(TypeError, match=error):
        relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=2.0,
            cosine_width=1.0,
        )


def test_relion_cuda_preprocess_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        cuda_backproject.relion_preprocess_real_f32.__wrapped__(
            jnp.zeros((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=2.0,
            cosine_width=1.0,
        )


def test_relion_cuda_preprocess_fails_closed_when_custom_cuda_disabled(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    with pytest.raises(RuntimeError, match="custom CUDA is disabled"):
        cuda_backproject.relion_preprocess_real_f32.__wrapped__(
            jnp.zeros((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=2.0,
            cosine_width=1.0,
        )
