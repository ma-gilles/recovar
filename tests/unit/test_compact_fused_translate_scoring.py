"""K-class compact pairs scored by the gather-free fused translate kernel.

The compact K>1 pass-2 path materializes ``(B, T, N)`` shifted images and two
``(B, P, N)`` pair gathers before RELION's diff2 accumulation. The fused
kernel translates each pixel inside the accumulation instead. These tests pin
that the routed result is bit-identical to the gathered production path and
that the route stays off unless explicitly enabled on a GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed as spb


def _compact_window(image_shape, current_size, *, include_nyquist_row):
    """Half-layout pixels of the full image inside the current-size crop."""
    n_rows, n_cols = image_shape
    half_width = n_cols // 2 + 1
    flat = np.arange(n_rows * half_width)
    ky = flat // half_width - n_rows // 2
    kx = flat % half_width
    low = -(current_size // 2) if include_nyquist_row else -(current_size // 2) + 1
    keep = (ky >= low) & (ky < current_size // 2) & (kx < current_size // 2 + 1)
    return flat[keep].astype(np.int32)


def test_fused_translate_route_is_off_by_default_and_fails_closed(monkeypatch):
    env = spb._SPARSE_KCLASS_COMPACT_FUSED_TRANSLATE_ENV
    angles = np.zeros((3, 2), dtype=np.float32)
    kwargs = dict(
        use_exact_relion_gaussian=True,
        use_float64_scoring=False,
        relion_score_translation_angles=angles,
        current_size=64,
    )
    monkeypatch.delenv(env, raising=False)
    assert spb._compact_fused_translate_scoring_enabled(**kwargs) is False
    monkeypatch.setenv(env, "1")
    assert (
        spb._compact_fused_translate_scoring_enabled(
            **dict(kwargs, use_float64_scoring=True)
        )
        is False
    )
    assert (
        spb._compact_fused_translate_scoring_enabled(
            **dict(kwargs, use_exact_relion_gaussian=False)
        )
        is False
    )
    assert (
        spb._compact_fused_translate_scoring_enabled(
            **dict(kwargs, relion_score_translation_angles=None)
        )
        is False
    )
    assert (
        spb._compact_fused_translate_scoring_enabled(**dict(kwargs, current_size=63))
        is False
    )
    if jax.default_backend() != "gpu":
        assert spb._compact_fused_translate_scoring_enabled(**kwargs) is False


@pytest.mark.gpu
@pytest.mark.parametrize("include_nyquist_row", [False, True])
def test_fused_translate_compact_pairs_match_gathered_path_bitwise(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    include_nyquist_row,
):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(20260912 + int(include_nyquist_row))
    image_shape = (32, 32)
    current_size = 24
    window = _compact_window(
        image_shape, current_size, include_nyquist_row=include_nyquist_row
    )
    n_pixels = int(window.size)
    batch, n_rows, n_trans, n_pairs = 3, 5, 4, 9
    fine_translations = rng.uniform(-2.5, 2.5, (n_trans, 2))
    angles = spb._relion_cuda_score_translation_angles_if_available(
        fine_translations, image_shape, enabled=True, dtype=np.float32
    )
    assert angles is not None and angles.dtype == np.float32
    lookup = spb._relion_cuda_fine_full_to_compact_lookup(
        image_shape, current_size, window
    )
    image = (
        rng.normal(0, 0.03, (batch, n_pixels))
        + 1j * rng.normal(0, 0.03, (batch, n_pixels))
    ).astype(np.complex64)
    proj = (
        rng.normal(0, 0.03, (batch, n_rows, n_pixels))
        + 1j * rng.normal(0, 0.03, (batch, n_rows, n_pixels))
    ).astype(np.complex64)
    corr_img = rng.uniform(0, 2.0e5, (batch, n_pixels)).astype(np.float32)
    half_weights = rng.choice([1.0, 2.0], size=n_pixels).astype(np.float32)
    rotation_rows = rng.integers(0, n_rows, (batch, n_pairs)).astype(np.int32)
    translation_ids = rng.integers(0, n_trans, (batch, n_pairs)).astype(np.int32)
    pair_mask = rng.uniform(size=(batch, n_pairs)) < 0.7
    pair_mask[:, 0] = True
    pair_mask[1, -1] = False
    highres = rng.uniform(0, 5, (batch,)).astype(np.float32)

    with jax.default_device(gpu_device):
        shifted = cuda_backproject.relion_translate_score_f32(
            jnp.asarray(image),
            jnp.asarray(angles),
            jnp.asarray(window),
            image_shape,
        ).reshape(batch, n_trans, n_pixels)
        gathered_jax = spb._score_pass2_pairs_relion_gpu_diff2_raw(
            shifted, corr_img, proj, half_weights, rotation_rows, translation_ids,
            pair_mask, lookup, highres, use_fused_ffi=False,
        )
        gathered_ffi = spb._score_pass2_pairs_relion_gpu_diff2_raw(
            shifted, corr_img, proj, half_weights, rotation_rows, translation_ids,
            pair_mask, lookup, highres, use_fused_ffi=True,
        )
        fused = spb._score_pass2_pairs_relion_gpu_diff2_raw_fused_translate(
            image, corr_img, proj, half_weights, angles, rotation_rows,
            translation_ids, pair_mask, lookup, highres, current_size=current_size,
        )
        gathered_jax, gathered_ffi, fused = jax.block_until_ready(
            (gathered_jax, gathered_ffi, fused)
        )
    gathered_jax = np.asarray(gathered_jax)
    gathered_ffi = np.asarray(gathered_ffi)
    fused = np.asarray(fused)
    assert fused.dtype == np.float32 and fused.shape == (batch, n_pairs)
    assert np.all(np.isfinite(fused))
    np.testing.assert_array_equal(
        gathered_ffi.view(np.uint32), gathered_jax.view(np.uint32)
    )
    np.testing.assert_array_equal(fused.view(np.uint32), gathered_jax.view(np.uint32))
