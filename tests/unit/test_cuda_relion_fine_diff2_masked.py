"""Masked rectangular fine diff2: RELION-style pair pruning inside the kernel."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from helpers.cuda_source import read_cuda_source


def _operands(rng, batch_size, rotation_count, translation_count, compact, full):
    reference = (
        rng.normal(0, 0.02, (batch_size, rotation_count, compact))
        + 1j * rng.normal(0, 0.02, (batch_size, rotation_count, compact))
    ).astype(np.complex64)
    shifted = (
        rng.normal(0, 0.02, (batch_size, translation_count, compact))
        + 1j * rng.normal(0, 0.02, (batch_size, translation_count, compact))
    ).astype(np.complex64)
    weight = rng.uniform(0, 150_000, (batch_size, compact)).astype(np.float32)
    initial = rng.uniform(10_000, 20_000, batch_size).astype(np.float32)
    retained = np.sort(rng.choice(full, compact, replace=False))
    lookup = np.full(full, -1, dtype=np.int32)
    lookup[retained] = np.arange(compact, dtype=np.int32)
    return reference, shifted, weight, initial, lookup


def test_masked_fine_diff2_source_pins_rectangular_body_and_zero_fill():
    source = read_cuda_source("../em/cuda/relion_scoring.cuh")
    start = source.index("void relion_fine_diff2_rectangular_masked_kernel(")
    body = source[start : source.index("cudaError_t launch_relion_fine_diff2_rectangular_masked(")]
    assert "if (candidate_mask[hypothesis] == 0) {" in body
    assert "output[hypothesis] = static_cast<T>(0);" in body
    # Same production lane topology and rounding boundaries as the unmasked kernel.
    assert body.count("relion_fine_diff2_update_f32(") == 1
    assert "__fadd_rn(lane_sums[0], initial_diff2[batch])" in body
    assert "kRelionFineDiff2BlockSize / 2; width > 0; width /= 2" in body
    handler = read_cuda_source("cuda_backproject.cu")
    assert "RelionFineDiff2RectangularMaskedF32Impl" in handler
    assert "candidate_mask.element_type() != ffi::DataType::PRED" in handler


def test_masked_fine_diff2_is_optional_ffi_target():
    from recovar import cuda_backproject

    target = cuda_backproject._TARGET_RELION_FINE_DIFF2_RECTANGULAR_MASKED_F32
    assert target == "cuda_relion_fine_diff2_rectangular_masked_f32"
    assert target in cuda_backproject._OPTIONAL_FFI_REGISTRATIONS
    assert target not in dict(cuda_backproject._FFI_REGISTRATIONS)


def test_fine_diff2_masked_knob_defaults_on_and_is_disableable(monkeypatch):
    """Default on after the matched hp3 pair (job 14082785); still switchable."""
    from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring

    monkeypatch.delenv(scoring._RELION_FINE_DIFF2_MASKED_ENV, raising=False)
    assert scoring._fine_diff2_masked_enabled() is True
    monkeypatch.setenv(scoring._RELION_FINE_DIFF2_MASKED_ENV, "0")
    assert scoring._fine_diff2_masked_enabled() is False


def test_masked_fine_diff2_falls_back_when_library_lacks_the_target(monkeypatch):
    """A library without the optional symbol must not break the default path."""
    import numpy as np

    from recovar.em.cuda import kernels as em_cuda_kernels
    from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring

    monkeypatch.setattr(
        em_cuda_kernels, "relion_fine_diff2_rectangular_masked_supported", lambda: False
    )
    called = {}

    def _unmasked(*args, **kwargs):
        called["unmasked"] = True
        return jnp.zeros((args[0].shape[0], args[0].shape[1], args[1].shape[1]), jnp.float32)

    def _masked(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("masked target used although the library lacks it")

    monkeypatch.setattr(em_cuda_kernels, "relion_fine_diff2_rectangular_f32", _unmasked)
    monkeypatch.setattr(
        em_cuda_kernels, "relion_fine_diff2_rectangular_masked_f32", _masked
    )
    rng = np.random.default_rng(5)
    reference, shifted, weight, _initial, lookup = _operands(rng, 2, 3, 4, 17, 21)
    scoring._relion_cuda_fine_diff2_sum(
        jnp.asarray(reference)[:, :, None, :],
        jnp.asarray(shifted)[:, None, :, :],
        jnp.asarray(weight)[:, None, None, :],
        jnp.asarray(lookup),
        use_fused_ffi=True,
        candidate_mask=jnp.asarray(rng.uniform(size=(2, 3, 4)) < 0.5),
    )
    assert called.get("unmasked") is True


def test_masked_support_probe_reports_false_without_a_library(monkeypatch):
    from recovar import cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(
        cuda_backproject, "_ensure_ffi", lambda: (_ for _ in ()).throw(RuntimeError("no lib"))
    )
    monkeypatch.setattr(
        em_cuda_kernels, "_ensure_ffi", lambda: (_ for _ in ()).throw(RuntimeError("no lib"))
    )
    assert em_cuda_kernels.relion_fine_diff2_rectangular_masked_supported() is False


def test_fine_diff2_sum_jax_path_ignores_candidate_mask(monkeypatch):
    """Without the fused FFI the emulation evaluates every cell (mask is a no-op)."""
    from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring

    monkeypatch.delenv(scoring._RELION_FINE_DIFF2_FUSED_FFI_ENV, raising=False)
    rng = np.random.default_rng(3)
    reference, shifted, weight, _initial, lookup = _operands(rng, 2, 3, 4, 17, 21)
    mask = rng.uniform(size=(2, 3, 4)) < 0.5
    with jax.default_device(jax.devices("cpu")[0]):
        plain = scoring._relion_cuda_fine_diff2_sum(
            jnp.asarray(reference)[:, :, None, :],
            jnp.asarray(shifted)[:, None, :, :],
            jnp.asarray(weight)[:, None, None, :],
            jnp.asarray(lookup),
        )
        masked = scoring._relion_cuda_fine_diff2_sum(
            jnp.asarray(reference)[:, :, None, :],
            jnp.asarray(shifted)[:, None, :, :],
            jnp.asarray(weight)[:, None, None, :],
            jnp.asarray(lookup),
            candidate_mask=jnp.asarray(mask),
        )
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(masked))


@pytest.mark.gpu
def test_masked_fine_diff2_matches_rectangular_on_valid_cells(
    monkeypatch,
    custom_cuda_lib,
    gpu_device,
):
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(29)
    batch_size, rotation_count, translation_count = 3, 17, 29
    compact, full = 421, 513
    reference, shifted, weight, initial, lookup = _operands(
        rng, batch_size, rotation_count, translation_count, compact, full
    )
    mask = rng.uniform(size=(batch_size, rotation_count, translation_count)) < 0.15
    mask[0, 0, :] = False  # a fully excluded row
    mask[1, :, 0] = True  # a fully included translation column
    with jax.default_device(gpu_device):
        dense = np.asarray(
            em_cuda_kernels.relion_fine_diff2_rectangular_f32(
                jnp.asarray(reference),
                jnp.asarray(shifted),
                jnp.asarray(weight),
                jnp.asarray(lookup),
                jnp.asarray(initial),
            )
        )
        masked = np.asarray(
            em_cuda_kernels.relion_fine_diff2_rectangular_masked_f32(
                jnp.asarray(reference),
                jnp.asarray(shifted),
                jnp.asarray(weight),
                jnp.asarray(lookup),
                jnp.asarray(mask),
                jnp.asarray(initial),
            )
        )
    assert masked.shape == dense.shape == (batch_size, rotation_count, translation_count)
    np.testing.assert_array_equal(masked[mask], dense[mask])
    assert np.all(masked[~mask] == 0.0)
    assert np.all(np.isfinite(dense)) and np.all(dense[mask] > 0)


@pytest.mark.gpu
def test_scoring_path_masked_matches_unmasked_scores(monkeypatch, custom_cuda_lib, gpu_device):
    """The full raw->scores path gives identical scores with the mask on and off."""
    import recovar.cuda_backproject as cuda_backproject
    from recovar.em.sparse_pass2 import sparse_pass2_scoring as scoring

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    rng = np.random.default_rng(11)
    batch_size, rotation_count, translation_count = 2, 9, 13
    compact, full = 201, 257
    reference, shifted, weight, initial, lookup = _operands(
        rng, batch_size, rotation_count, translation_count, compact, full
    )
    half_weights = rng.uniform(0.5, 1.0, compact).astype(np.float32)
    mask = rng.uniform(size=(batch_size, rotation_count, translation_count)) < 0.3
    mask[:, 0, 0] = True
    rot_prior = rng.normal(0, 1, (batch_size, rotation_count)).astype(np.float32)
    trans_prior = rng.normal(0, 1, (batch_size, translation_count)).astype(np.float32)
    with jax.default_device(gpu_device):
        args = (
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(reference),
            jnp.asarray(half_weights),
            jnp.asarray(lookup),
            jnp.asarray(initial),
        )
        plain_raw = scoring._score_pass2_bucket_relion_gpu_diff2_raw(*args, use_fused_ffi=True)
        masked_raw = scoring._score_pass2_bucket_relion_gpu_diff2_raw(
            *args, use_fused_ffi=True, candidate_mask=jnp.asarray(mask)
        )
        m = jnp.asarray(mask)
        plain_min = scoring._relion_cuda_fine_diff2_min(plain_raw, m)
        masked_min = scoring._relion_cuda_fine_diff2_min(masked_raw, m)
        plain_scores = scoring._relion_cuda_fine_diff2_to_scores(
            plain_raw, jnp.asarray(rot_prior)[:, :, None], jnp.asarray(trans_prior)[:, None, :], m, min_diff2=plain_min
        )
        masked_scores = scoring._relion_cuda_fine_diff2_to_scores(
            masked_raw, jnp.asarray(rot_prior)[:, :, None], jnp.asarray(trans_prior)[:, None, :], m, min_diff2=masked_min
        )
    np.testing.assert_array_equal(np.asarray(plain_min), np.asarray(masked_min))
    np.testing.assert_array_equal(np.asarray(plain_scores), np.asarray(masked_scores))
    np.testing.assert_array_equal(np.asarray(plain_raw)[mask], np.asarray(masked_raw)[mask])
