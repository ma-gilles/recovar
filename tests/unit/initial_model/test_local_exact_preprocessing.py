from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.helpers.half_spectrum import make_half_image_weights
from recovar.em.local import local_preprocessing
from recovar.em.local.local_caches import _LocalProcessedHalfCache
from recovar.em.relion import relion_ctf

pytestmark = pytest.mark.unit


class _CapturedPreprocess(RuntimeError):
    pass


def _exercise_prepare(monkeypatch: pytest.MonkeyPatch, *, backend: str):
    captured = {}

    def capture_process(_dataset, batch, apply_image_mask, *, relion_preprocess_kwargs=None):
        captured.update(
            batch=np.asarray(batch),
            apply_image_mask=apply_image_mask,
            kwargs=relion_preprocess_kwargs,
        )
        raise _CapturedPreprocess

    monkeypatch.setattr(local_preprocessing, "process_half_image", capture_process)
    dataset = SimpleNamespace(
        image_source=SimpleNamespace(backend=SimpleNamespace(relion_fourier_backend=backend)),
    )
    config = SimpleNamespace(
        image_shape=(4, 4),
        compute_ctf_half=lambda params: jnp.ones((len(params), 12), dtype=jnp.float32),
    )
    images = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    with pytest.raises(_CapturedPreprocess):
        local_preprocessing.prepare_local_bucket(
            dataset,
            images,
            np.zeros((2, 9), dtype=np.float32),
            np.asarray([3, 7], dtype=np.int32),
            jnp.ones(12, dtype=jnp.float32),
            jnp.ones((1, 12), dtype=jnp.complex64),
            config,
            jnp.ones(12, dtype=jnp.float32),
            True,
        )
    return images, captured


@pytest.mark.unit
def test_split_local_exact_supplies_identity_operands_to_relion_cuda(monkeypatch):
    images, captured = _exercise_prepare(monkeypatch, backend="relion_cuda")

    np.testing.assert_array_equal(captured["batch"], images)
    np.testing.assert_array_equal(captured["kwargs"]["relion_normalization_factors"], np.ones(2, np.float32))
    np.testing.assert_array_equal(captured["kwargs"]["relion_integer_shifts"], np.zeros((2, 2), np.int32))
    assert captured["apply_image_mask"] is True


@pytest.mark.unit
def test_split_local_exact_leaves_general_backend_operands_unset(monkeypatch):
    images, captured = _exercise_prepare(monkeypatch, backend="jax_gpu")

    np.testing.assert_array_equal(captured["batch"], images)
    assert captured["kwargs"] is None


@pytest.mark.parametrize("score_with_masked_images", [False, True])
def test_prepare_local_exact_bucket_preserves_relion_bpref_operand_orders(
    monkeypatch,
    score_with_masked_images,
):
    import recovar.cuda_backproject as cuda_backproject

    # This test exercises host operand assembly, not preprocessing or CUDA kernels.
    dataset = SimpleNamespace(image_shape=(8, 8))
    config = SimpleNamespace(image_shape=dataset.image_shape)
    n_half = dataset.image_shape[0] * (dataset.image_shape[1] // 2 + 1)
    processed = (
        np.arange(n_half, dtype=np.float32)[None, :]
        + 1j * np.arange(n_half, dtype=np.float32)[None, ::-1]
    ).astype(np.complex64)
    masked_processed = processed * np.float32(0.5)
    expected_score = masked_processed if score_with_masked_images else processed
    mask_calls = []

    def fake_preprocess(_batch, _mask, _config, *, apply_image_mask, mask_mode):
        mask_calls.append(apply_image_mask)
        return jnp.asarray(masked_processed if apply_image_mask else processed)

    ctf_rfloat = np.linspace(-1.25, 1.75, n_half, dtype=np.float64)[None, :]
    noise_f64 = np.linspace(0.7, 2.3, n_half, dtype=np.float64)

    monkeypatch.setattr(
        relion_ctf,
        "_relion_exact_ctf_half_from_source_star_host",
        lambda *_args, **_kwargs: ctf_rfloat,
    )
    monkeypatch.setattr(
        local_preprocessing,
        "resolve_image_mask_for_half_preprocess",
        lambda *_args, **_kwargs: (np.zeros(dataset.image_shape, dtype=np.float32), "none"),
    )
    monkeypatch.setattr(
        local_preprocessing,
        "_big_jit_preprocess_half",
        fake_preprocess,
    )
    captured_score = {}

    def fake_translate_score(images, translation_angles, pixel_indices, image_shape):
        captured_score.update(
            images=np.asarray(images),
            translation_angles=np.asarray(translation_angles),
            pixel_indices=np.asarray(pixel_indices),
            image_shape=tuple(image_shape),
        )
        # The requested translation is exactly zero; its phase is identity.
        return images[:, None, :].reshape(-1, images.shape[1])

    monkeypatch.setattr(cuda_backproject, "relion_translate_score_f32", fake_translate_score)
    captured_bpref = {}

    def fake_relion_translate_bpref(
        images,
        weighted_ctf,
        translation_angles,
        pixel_indices,
        image_shape,
    ):
        captured_bpref.update(
            images=np.asarray(images),
            weighted_ctf=np.asarray(weighted_ctf),
            translation_angles=np.asarray(translation_angles),
            pixel_indices=np.asarray(pixel_indices),
            image_shape=tuple(image_shape),
        )
        return (images[:, None, :] * weighted_ctf[:, None, :]).reshape(-1, images.shape[1])

    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_bpref_f32",
        fake_relion_translate_bpref,
    )

    (
        shifted_score,
        shifted_recon,
        batch_norm,
        score_weight,
        recon_weight,
        processed_score,
        pre_shift_applied,
    ) = local_preprocessing.prepare_local_bucket(
        dataset,
        np.zeros((1, *dataset.image_shape), dtype=np.float32),
        np.zeros((1, 9), dtype=np.float32),
        np.asarray([0], dtype=np.int32),
        jnp.asarray(noise_f64),
        jnp.zeros((1, 2), dtype=jnp.float32),
        config,
        make_half_image_weights(dataset.image_shape),
        score_with_masked_images=score_with_masked_images,
        relion_score_translation_angles=np.zeros((1, 2), dtype=np.float32),
        relion_exact_bpref_operands=True,
    )

    inverse_noise = np.reciprocal(noise_f64).astype(np.float32)
    ctf_f32 = ctf_rfloat.astype(np.float32)
    expected_weighted_ctf = ctf_f32 * inverse_noise[None, :]
    expected_score_weight = (
        inverse_noise[None, :].astype(np.float64) * ctf_rfloat * ctf_rfloat
    ).astype(np.float32)
    expected_recon_weight = expected_weighted_ctf * ctf_f32
    np.testing.assert_array_equal(np.asarray(processed_score), expected_score)
    np.testing.assert_array_equal(np.asarray(shifted_score), expected_score * expected_weighted_ctf)
    np.testing.assert_array_equal(np.asarray(shifted_recon), processed * expected_weighted_ctf)
    np.testing.assert_array_equal(np.asarray(score_weight), expected_score_weight)
    np.testing.assert_array_equal(np.asarray(recon_weight), expected_recon_weight)
    np.testing.assert_array_equal(captured_bpref["images"], processed)
    np.testing.assert_array_equal(captured_bpref["weighted_ctf"], expected_weighted_ctf)
    np.testing.assert_array_equal(captured_bpref["translation_angles"], np.zeros((1, 2), np.float32))
    np.testing.assert_array_equal(captured_bpref["pixel_indices"], np.arange(n_half, dtype=np.int32))
    assert captured_bpref["image_shape"] == dataset.image_shape
    expected_norm = np.sum(
        np.abs(expected_score) ** 2
        * inverse_noise[None, :]
        * np.asarray(make_half_image_weights(dataset.image_shape))[None, :],
        axis=-1,
        keepdims=True,
    )
    np.testing.assert_allclose(np.asarray(batch_norm), expected_norm, rtol=1e-6, atol=1e-5)
    assert pre_shift_applied is False
    assert mask_calls == ([True, False] if score_with_masked_images else [False])
    np.testing.assert_array_equal(captured_score["images"], expected_score * expected_weighted_ctf)
    np.testing.assert_array_equal(captured_score["translation_angles"], np.zeros((1, 2), np.float32))
    np.testing.assert_array_equal(captured_score["pixel_indices"], np.arange(n_half, dtype=np.int32))
    assert captured_score["image_shape"] == dataset.image_shape


@pytest.fixture
def cached_bucket(monkeypatch):
    """Nontrivial image order distinguishes cache indexing from prefix slicing."""
    score = np.arange(48, dtype=np.float32).reshape(4, 12).astype(np.complex64)
    recon = score + np.complex64(100 + 2j)
    cache = _LocalProcessedHalfCache(
        ctf_params=np.zeros((4, 9), dtype=np.float32),
        score_half=score,
        recon_half=recon,
        integer_pre_shifts_applied=True,
    )

    def unexpected_preprocess(*args, **kwargs):
        pytest.fail("A populated processed cache must bypass image preprocessing")

    monkeypatch.setattr(local_preprocessing, "process_half_image", unexpected_preprocess)
    return dict(
        experiment_dataset=SimpleNamespace(),
        batch=np.zeros((2, 4, 4), dtype=np.float32),
        ctf_params=cache.ctf_params[[3, 1]],
        image_indices=np.array([3, 1], dtype=np.int32),
        noise_variance_half=jnp.ones(12, dtype=jnp.float32),
        translation_phases_half=jnp.ones((1, 12), dtype=jnp.complex64),
        config=SimpleNamespace(
            image_shape=(4, 4),
            compute_ctf_half=lambda params: jnp.ones((len(params), 12), dtype=jnp.float32),
        ),
        norm_half_weights=jnp.ones(12, dtype=jnp.float32),
        processed_half_cache=cache,
    )


@pytest.mark.parametrize("score_with_masked_images", [False, True])
def test_cached_bucket_preserves_image_order_and_reconstruction_source(cached_bucket, score_with_masked_images):
    result = local_preprocessing.prepare_local_bucket(
        **cached_bucket, score_with_masked_images=score_with_masked_images,
    )
    shifted_score, shifted_recon, _, _, _, processed_score, pre_shift_applied = result
    cache = cached_bucket["processed_half_cache"]
    indices = cached_bucket["image_indices"]
    expected_score = cache.score_half[indices]
    expected_recon = cache.recon_half[indices] if score_with_masked_images else expected_score
    np.testing.assert_array_equal(processed_score, expected_score)
    np.testing.assert_array_equal(shifted_score, expected_score)
    np.testing.assert_array_equal(shifted_recon, expected_recon)
    assert pre_shift_applied is True
    if not score_with_masked_images:
        assert shifted_recon is shifted_score


def test_masked_cached_bucket_requires_unmasked_reconstruction(cached_bucket):
    cache = cached_bucket["processed_half_cache"]
    cached_bucket["processed_half_cache"] = _LocalProcessedHalfCache(
        ctf_params=cache.ctf_params,
        score_half=cache.score_half,
        recon_half=None,
        integer_pre_shifts_applied=True,
    )
    with pytest.raises(RuntimeError, match="^processed half-image cache is missing unmasked reconstruction images$"):
        local_preprocessing.prepare_local_bucket(**cached_bucket, score_with_masked_images=True)

    # The same cache is valid when reconstruction reuses unmasked score images.
    result = local_preprocessing.prepare_local_bucket(**cached_bucket, score_with_masked_images=False)
    assert result[1] is result[0]
