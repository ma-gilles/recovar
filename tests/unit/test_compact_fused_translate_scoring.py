"""Compact fused scoring: exact CUDA parity, masking, grouping and dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.sparse_pass2 import sparse_pass2_scoring as spb
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import _relion_cuda_score_translation_angles_if_available


def _compact_window(image_shape, current_size):
    """Half-layout pixels of the current-size crop, in production row order.

    ``_relion_half_layout_mask`` keeps rows ``ky = -(cs/2 - 1) .. cs/2``: the
    FFTW row ``cs/2`` is the positive Nyquist frequency, which is what both
    RELION's fine kernel (``y > maxR`` wrap, ``maxR = cs/2``) and the fused
    kernel use for the translation phase. The ``ky = -cs/2`` row is never in
    a production window; with it the two conventions would disagree.
    """
    n_rows, n_cols = image_shape
    half_width = n_cols // 2 + 1
    flat = np.arange(n_rows * half_width)
    ky = flat // half_width - n_rows // 2
    kx = flat % half_width
    keep = (ky >= -(current_size // 2) + 1) & (ky <= current_size // 2) & (kx <= current_size // 2)
    return flat[keep].astype(np.int32)


def _ulp_distance(a, b):
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))


@pytest.mark.unit
@pytest.mark.parametrize("backend,custom_cuda,expected", [("cpu", True, False), ("gpu", False, False), ("gpu", True, True)])
def test_fused_translate_route_requires_supported_gpu(monkeypatch, backend, custom_cuda, expected):
    from recovar import cuda_backproject

    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(cuda_backproject, "custom_cuda_requested", lambda: custom_cuda)
    kwargs = dict(
        use_exact_relion_gaussian=True,
        use_float64_scoring=False,
        relion_score_translation_angles=np.zeros((3, 2), dtype=np.float32),
        current_size=64,
    )
    assert spb._compact_fused_translate_scoring_enabled(**kwargs) is expected
    for change in (
        {"use_float64_scoring": True},
        {"use_exact_relion_gaussian": False},
        {"relion_score_translation_angles": None},
        {"current_size": None},
        {"current_size": 0},
        {"current_size": 63},
    ):
        assert spb._compact_fused_translate_scoring_enabled(**(kwargs | change)) is False


@pytest.mark.gpu
@pytest.mark.parametrize("translation_mode", ["zero", "random"])
@pytest.mark.parametrize("n_pairs", [1, 4, 7, 16, 17, 64])
def test_fused_translate_compact_pairs_match_gathered_path(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
    translation_mode,
    n_pairs,
):
    """Fused route == FFI pairs kernel bitwise; a few ULP from the JAX emulation.

    The pure-JAX 256-lane emulation used by the gathered path when
    ``RECOVAR_RELION_FINE_DIFF2_FUSED_FFI`` is off differs from the CUDA
    kernels by one or two binary32 ULP on a minority of pairs even with zero
    translations, so the kernel, not the emulation, is the bitwise reference.
    """
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(20260912 + (translation_mode == "random"))
    image_shape = (32, 32)
    current_size = 24
    window = _compact_window(image_shape, current_size)
    n_pixels = int(window.size)
    batch, n_rows, n_trans = 4, 6, 5
    fine_translations = (
        np.zeros((n_trans, 2))
        if translation_mode == "zero"
        else rng.uniform(-2.5, 2.5, (n_trans, 2))
    )
    angles = _relion_cuda_score_translation_angles_if_available(
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
    # masked pairs are skipped by the kernel and come back +inf (consumers treat
    # non-finite as invalid); valid pairs are bit-identical to the FFI kernel
    assert np.all(np.isposinf(fused[~pair_mask]))
    assert np.all(np.isfinite(fused[pair_mask]))
    np.testing.assert_array_equal(fused[pair_mask].view(np.uint32), gathered_ffi[pair_mask].view(np.uint32))
    # the emulation comparisons below are meaningful for valid pairs only
    fused = fused[pair_mask]; gathered_ffi = gathered_ffi[pair_mask]; gathered_jax = gathered_jax[pair_mask]
    # The emulation is not the bitwise reference: on H100 (jobs 13803046 and
    # 13803570) it sits 1-2 ULP from the CUDA kernels on a minority of pairs.
    # Bound it loosely so a real regression (many ULP) still fails here.
    assert int(_ulp_distance(fused, gathered_jax).max()) <= 4
    assert int(_ulp_distance(gathered_ffi, gathered_jax).max()) <= 4
