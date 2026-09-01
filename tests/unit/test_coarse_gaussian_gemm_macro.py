"""Focused CPU contracts for the shared coarse projection/GEMM macro."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers import scoring, significance


def _macro_operands(*, real_dtype, n_images=4, n_trans=3, n_rotations=5, n_pixels=11):
    rng = np.random.default_rng(20260901)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    projected = (
        rng.normal(size=(n_rotations, n_pixels))
        + 1j * rng.normal(size=(n_rotations, n_pixels))
    ).astype(complex_dtype)
    shifted = (
        rng.normal(size=(n_images, n_trans, n_pixels))
        + 1j * rng.normal(size=(n_images, n_trans, n_pixels))
    ).astype(complex_dtype)
    weight = rng.uniform(0.05, 2.0, size=(n_images, n_pixels)).astype(real_dtype)
    initial = rng.uniform(0.0, 4.0, size=n_images).astype(real_dtype)
    return projected, np.abs(projected) ** 2, shifted, weight, initial


def _direct_scores(projected, shifted, weight, initial):
    difference = projected[None, :, None, :] - shifted[:, None, :, :]
    return -initial[:, None, None] - 0.5 * np.sum(
        weight[:, None, None, :] * np.abs(difference) ** 2,
        axis=-1,
    )


def test_coarse_gaussian_gemm_macro_is_default_off_and_fail_closed(monkeypatch):
    variable = "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO"
    monkeypatch.delenv(variable, raising=False)
    assert not significance._coarse_gaussian_gemm_macro_enabled()
    assert significance._coarse_gaussian_gemm_macro_enabled(default=True)

    for disabled in ("0", "false", "no", "off"):
        monkeypatch.setenv(variable, disabled)
        assert not significance._coarse_gaussian_gemm_macro_enabled(default=True)
    for enabled in ("1", "true", "yes", "on"):
        monkeypatch.setenv(variable, enabled)
        assert significance._coarse_gaussian_gemm_macro_enabled()

    monkeypatch.setenv(variable, "automatic")
    with pytest.raises(ValueError, match=variable):
        significance._coarse_gaussian_gemm_macro_enabled()


def test_coarse_gaussian_gemm_scores_match_direct_objective_float32():
    """Float32 GEMM reduction agrees with the direct mathematical objective."""

    operands = _macro_operands(real_dtype=np.float32)
    actual = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, operands),
            operands[2].shape[0],
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    expected = _direct_scores(operands[0], operands[2], operands[3], operands[4])

    assert actual.shape == (4, 5, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        np.argmax(actual.reshape(actual.shape[0], -1), axis=1),
        np.argmax(expected.reshape(expected.shape[0], -1), axis=1),
    )


def test_coarse_gaussian_gemm_scores_match_direct_objective_float64():
    """Float64 companion tightens the float32 arithmetic envelope by >3 orders."""

    operands = _macro_operands(real_dtype=np.float64)
    actual = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, operands),
            operands[2].shape[0],
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    expected = _direct_scores(operands[0], operands[2], operands[3], operands[4])

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        np.argmax(actual.reshape(actual.shape[0], -1), axis=1),
        np.argmax(expected.reshape(expected.shape[0], -1), axis=1),
    )


def test_coarse_gaussian_gemm_scores_ignore_poisoned_tail_exactly():
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
        n_images=5,
    )
    actual_image_count = 3
    clean_shifted = shifted.copy()
    clean_weight = weight.copy()
    clean_initial = initial.copy()
    clean_shifted[actual_image_count:] = 0
    clean_weight[actual_image_count:] = 0
    clean_initial[actual_image_count:] = 0
    poisoned_shifted = clean_shifted.copy()
    poisoned_weight = clean_weight.copy()
    poisoned_initial = clean_initial.copy()
    poisoned_shifted[actual_image_count:] = np.complex64(np.nan + 1j * np.nan)
    poisoned_weight[actual_image_count:] = np.nan
    poisoned_initial[actual_image_count:] = np.nan

    common = (
        jnp.asarray(projected),
        jnp.asarray(projected_abs2),
    )
    clean = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *common,
            jnp.asarray(clean_shifted),
            jnp.asarray(clean_weight),
            jnp.asarray(clean_initial),
            actual_image_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )
    poisoned = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            *common,
            jnp.asarray(poisoned_shifted),
            jnp.asarray(poisoned_weight),
            jnp.asarray(poisoned_initial),
            actual_image_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
    )

    np.testing.assert_array_equal(
        poisoned[:actual_image_count].view(np.uint32),
        clean[:actual_image_count].view(np.uint32),
    )
    np.testing.assert_array_equal(
        poisoned[actual_image_count:].view(np.uint32),
        np.zeros_like(poisoned[actual_image_count:]).view(np.uint32),
    )


def test_coarse_gaussian_gemm_macro_projects_once_and_binds_all_image_lanes(
    monkeypatch,
):
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
        n_images=6,
        n_rotations=7,
    )
    rotations = np.arange(7 * 3 * 3, dtype=np.float32).reshape(7, 3, 3)
    mean = object()
    projection_calls = []
    score_calls = []

    def project_once(class_index, mean_for_proj, rotations_block):
        projection_calls.append((class_index, mean_for_proj, np.asarray(rotations_block)))
        return jnp.asarray(projected), jnp.asarray(projected_abs2)

    sentinel = jnp.arange(6 * 7 * 3, dtype=jnp.float32).reshape(6, 7, 3)

    def capture_score(*args, **kwargs):
        score_calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        significance,
        "_relion_coarse_gaussian_gemm_scores",
        capture_score,
    )
    result = significance._score_relion_coarse_gaussian_gemm_macro(
        project_once,
        2,
        mean,
        rotations,
        jnp.asarray(shifted),
        jnp.asarray(weight),
        jnp.asarray(initial),
        4,
        image_shape=(8, 8),
        volume_shape=(8, 8, 8),
    )

    assert result is sentinel
    assert len(projection_calls) == 1
    assert projection_calls[0][0] == 2
    assert projection_calls[0][1] is mean
    np.testing.assert_array_equal(projection_calls[0][2], rotations)
    assert len(score_calls) == 1
    bound, static = score_calls[0]
    np.testing.assert_array_equal(np.asarray(bound[0]), projected)
    np.testing.assert_array_equal(np.asarray(bound[1]), projected_abs2)
    np.testing.assert_array_equal(np.asarray(bound[2]), shifted)
    np.testing.assert_array_equal(np.asarray(bound[3]), weight)
    np.testing.assert_array_equal(np.asarray(bound[4]), initial)
    assert bound[5] == 4
    assert static == {"image_shape": (8, 8), "volume_shape": (8, 8, 8)}


def test_coarse_gaussian_gemm_macro_is_shared_by_em_and_initial_model():
    from recovar.em.dense_single_volume import k_class
    from recovar.em.initial_model import dense_adapter

    assert (
        dense_adapter._compute_k_class_significance_batched
        is significance._compute_k_class_significance_batched
    )
    em_source = inspect.getsource(k_class.run_dense_k_class_em_adaptive)
    assert "from .helpers.significance import _compute_k_class_significance_batched" in em_source
    assert "_compute_k_class_significance_batched(" in em_source
    initial_model_source = inspect.getsource(dense_adapter)
    assert "def _score_relion_coarse_gaussian_gemm_macro(" not in initial_model_source
    macro_score_source = inspect.getsource(
        scoring._relion_coarse_gaussian_gemm_scores_jit,
    )
    assert "_e_step_block_scores_windowed(" in macro_score_source
    assert "jnp.matmul(" not in macro_score_source


@pytest.mark.parametrize(
    ("bad_operand", "message"),
    [
        ("projection_pixels", "projection and image pixels"),
        ("weight_shape", "pixel_weight"),
        ("tail_count", "actual_image_count"),
    ],
)
def test_coarse_gaussian_gemm_macro_rejects_ambiguous_bindings(
    bad_operand,
    message,
):
    projected, projected_abs2, shifted, weight, initial = _macro_operands(
        real_dtype=np.float32,
    )
    actual_count = shifted.shape[0]
    if bad_operand == "projection_pixels":
        projected = projected[:, :-1]
        projected_abs2 = projected_abs2[:, :-1]
    elif bad_operand == "weight_shape":
        weight = weight[:, :-1]
    else:
        actual_count += 1

    with pytest.raises((ValueError, TypeError), match=message):
        scoring._relion_coarse_gaussian_gemm_scores(
            *map(jnp.asarray, (projected, projected_abs2, shifted, weight, initial)),
            actual_count,
            image_shape=(8, 8),
            volume_shape=(8, 8, 8),
        )
