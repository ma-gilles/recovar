from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume import local_em_engine


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

    monkeypatch.setattr(local_em_engine, "process_half_image", capture_process)
    dataset = SimpleNamespace(
        image_source=SimpleNamespace(backend=SimpleNamespace(relion_fourier_backend=backend)),
    )
    config = SimpleNamespace(
        image_shape=(4, 4),
        compute_ctf_half=lambda params: jnp.ones((len(params), 12), dtype=jnp.float32),
    )
    images = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    with pytest.raises(_CapturedPreprocess):
        local_em_engine._prepare_local_exact_bucket(
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
