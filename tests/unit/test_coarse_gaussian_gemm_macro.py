"""Focused CPU contracts for the shared coarse projection/GEMM macro."""

from __future__ import annotations

import jax
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


def _expanded_square_roundoff_bound(projected, shifted, weight):
    """Conservative arithmetic bound, not a production acceptance tolerance."""

    real_dtype = np.asarray(weight).dtype
    eps = np.finfo(real_dtype).eps
    n_pixels = int(projected.shape[-1])
    # Complex products, two dot products, and the image-power reduction each
    # contribute multiple rounded operations.  This deliberately conservative
    # gamma bound only asserts finite/bounded arithmetic in this CPU stress
    # test; it is not used by production selection or promotion.
    gamma = (32 * n_pixels * eps) / (1.0 - min(0.5, 32 * n_pixels * eps))
    model = np.sum(
        weight[:, None, :] * np.abs(projected[None, :, :]) ** 2,
        axis=-1,
        dtype=real_dtype,
    )
    cross = 2.0 * np.sum(
        weight[:, None, None, :]
        * np.abs(shifted[:, None, :, :])
        * np.abs(projected[None, :, None, :]),
        axis=-1,
        dtype=real_dtype,
    )
    image = np.sum(
        weight[:, None, :] * np.abs(shifted) ** 2,
        axis=-1,
        dtype=real_dtype,
    )
    magnitude = model[..., None] + cross + image[:, None, :]
    return gamma * np.maximum(magnitude, np.asarray(1.0, dtype=real_dtype))


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


def _backend_kwargs(**updates):
    values = dict(
        gemm_macro_requested=True,
        score_mode="gaussian",
        fused_projector_requested=False,
        fused_projector_enabled=False,
        canonical_reduction_requested=False,
        canonical_reduction_enabled=False,
        native_atomic_reduction_requested=False,
        native_atomic_reduction_enabled=False,
        single_lane_canonical_requested=False,
        single_lane_canonical_enabled=False,
        multistream_requested=False,
        multistream_enabled=False,
        native_texture_requested=False,
        native_texture_enabled=False,
    )
    values.update(updates)
    return values


def test_coarse_gaussian_gemm_backend_resolves_only_when_unambiguous():
    assert significance._resolve_coarse_gaussian_score_backend(
        **_backend_kwargs(),
    ) is significance._CoarseGaussianScoreBackend.GEMM_MACRO


def test_coarse_gaussian_gemm_backend_rejects_non_gaussian_score_mode():
    with pytest.raises(ValueError, match="score_mode='gaussian'"):
        significance._resolve_coarse_gaussian_score_backend(
            **_backend_kwargs(score_mode="normalized_cc"),
        )


@pytest.mark.parametrize(
    ("selector", "environment_name"),
    [
        ("fused_projector_requested", "RECOVAR_K1_COARSE_FUSED_PROJECTOR"),
        ("canonical_reduction_requested", "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION"),
        ("native_atomic_reduction_requested", "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION"),
        ("single_lane_canonical_requested", "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL"),
        ("multistream_requested", "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS"),
        ("native_texture_requested", "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE"),
    ],
)
def test_coarse_gaussian_gemm_backend_rejects_every_competing_selector(
    selector,
    environment_name,
):
    with pytest.raises(ValueError, match=environment_name):
        significance._resolve_coarse_gaussian_score_backend(
            **_backend_kwargs(**{selector: True}),
        )


def test_coarse_gaussian_gemm_resource_gate_records_full_transient_and_host_sync():
    resources = significance._coarse_gaussian_gemm_resources(
        rotation_block_size=17,
        image_shape=(128, 128),
        compact_pixel_count=64 * 33,
        budget_bytes=10**9,
    )
    assert resources.full_centered_projection_bytes == 17 * 128 * 65 * 8
    assert resources.compact_projection_bytes == 17 * 64 * 33 * 8
    assert resources.compact_projection_abs2_bytes == 17 * 64 * 33 * 4
    assert resources.predicted_peak_projection_bytes == (
        resources.full_centered_projection_bytes
        + resources.compact_projection_bytes
        + resources.compact_projection_abs2_bytes
    )
    assert resources.pixel_index_device_to_host_materializations == 0

    with pytest.raises(MemoryError, match="predicted projection transient"):
        significance._coarse_gaussian_gemm_resources(
            rotation_block_size=17,
            image_shape=(128, 128),
            compact_pixel_count=64 * 33,
            budget_bytes=resources.predicted_peak_projection_bytes - 1,
        )


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


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 100.0, 1.0e4])
def test_coarse_gaussian_gemm_direct_square_cancellation_stress_equal_operands(
    real_dtype,
    scale,
):
    """p == s stays finite and inside an arithmetic bound at three scales."""

    if real_dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(13290001)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    projected = (
        scale
        * (
            rng.normal(size=(1, 4096))
            + 1j * rng.normal(size=(1, 4096))
        )
    ).astype(complex_dtype)
    shifted = projected[None, :, :].copy()
    weight = rng.uniform(0.5, 1.5, size=(1, 4096)).astype(real_dtype)
    initial = np.zeros(1, dtype=real_dtype)
    actual = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            jnp.asarray(projected),
            jnp.asarray(np.abs(projected) ** 2, dtype=real_dtype),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial),
            1,
            image_shape=(128, 128),
            volume_shape=(128, 128, 128),
        )
    )
    direct = _direct_scores(projected, shifted, weight, initial)
    signed_residual = -2.0 * actual
    bound = _expanded_square_roundoff_bound(projected, shifted, weight)

    np.testing.assert_array_equal(direct, np.zeros_like(direct))
    assert np.all(np.isfinite(signed_residual))
    assert np.all(np.abs(actual - direct) <= bound)


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_coarse_gaussian_gemm_direct_square_cancellation_stress_nearby_operands(
    real_dtype,
):
    """Production-scale p≈s exposes bounded signed expansion error."""

    if real_dtype == np.float64:
        jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(13290002)
    complex_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    projected = (
        100.0
        * (
            rng.normal(size=(3, 4096))
            + 1j * rng.normal(size=(3, 4096))
        )
    ).astype(complex_dtype)
    perturbation_scale = 1.0e-4 if real_dtype == np.float32 else 1.0e-8
    shifted = (
        projected[None, :1, :]
        + perturbation_scale
        * 100.0
        * (
            rng.normal(size=(1, 2, 4096))
            + 1j * rng.normal(size=(1, 2, 4096))
        )
    ).astype(complex_dtype)
    weight = rng.uniform(0.5, 1.5, size=(1, 4096)).astype(real_dtype)
    initial = np.zeros(1, dtype=real_dtype)
    macro = np.asarray(
        scoring._relion_coarse_gaussian_gemm_scores(
            jnp.asarray(projected),
            jnp.asarray(np.abs(projected) ** 2, dtype=real_dtype),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(initial),
            1,
            image_shape=(128, 128),
            volume_shape=(128, 128, 128),
        )
    )
    direct = _direct_scores(projected, shifted, weight, initial)
    delta = macro - direct
    bound = _expanded_square_roundoff_bound(projected, shifted, weight)

    assert np.all(np.isfinite(macro))
    assert np.any(delta > 0.0) or np.any(delta < 0.0)
    assert np.all(np.abs(delta) <= bound)


def test_coarse_gaussian_direct_macro_diagnostics_preserve_layout_and_discretes():
    direct = np.array(
        [[[[3.0, 1.0], [2.0, 0.0]]], [[[1.0, 4.0], [3.0, 2.0]]]],
        dtype=np.float32,
    )
    macro = direct.copy()
    macro[1, 0, 1, 0] = 5.0
    direct_support = np.array([[True, False, True, False], [False, True, True, False]])
    macro_support = direct_support.copy()
    macro_support[1, 2] = False
    diagnostics = scoring._coarse_gaussian_direct_macro_diagnostics(
        direct,
        macro,
        direct_support=direct_support,
        macro_support=macro_support,
    )

    assert diagnostics["score_delta"].shape == direct.shape
    np.testing.assert_array_equal(diagnostics["argmax_equal"], [True, False])
    np.testing.assert_array_equal(diagnostics["support_equal"], [True, False])
    np.testing.assert_array_equal(
        diagnostics["support_symmetric_difference_count"],
        [0, 1],
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
    assert (
        "_compute_k_class_significance_batched"
        in k_class.run_dense_k_class_em_adaptive.__code__.co_names
    )
    assert (
        scoring._relion_coarse_gaussian_gemm_scores_jit._fun.__globals__[
            "_e_step_block_scores_windowed"
        ]
        is scoring._e_step_block_scores_windowed
    )


class _MacroIntegrationDataset:
    """Tiny strict-preprocess dataset for the live shared significance path."""

    image_shape = (4, 4)
    image_size = 16
    grid_size = 4
    padding = 0
    volume_shape = (4, 4, 4)
    volume_size = 64
    voxel_size = 1.0
    dtype = jnp.complex64
    premultiplied_ctf = False

    def __init__(self):
        self.n_images = 3
        self.n_units = 3
        self._images = np.stack(
            [np.full(self.image_shape, value, dtype=np.float32) for value in (1, 2, 3)]
        )
        self.CTF_params = np.zeros((self.n_units, 9), dtype=np.float32)
        self.rotation_matrices = np.tile(np.eye(3, dtype=np.float32), (self.n_units, 1, 1))
        self.translations = np.zeros((self.n_units, 2), dtype=np.float32)

        class _Backend:
            image_mask = np.ones((4, 4), dtype=np.float32)
            image_mask_mode = "relion_background_fill"
            relion_fourier_backend = "relion_cuda"

        class _ImageSource:
            backend = _Backend()

        self.image_source = _ImageSource()

    @staticmethod
    def ctf_evaluator(params, image_shape=None, voxel_size=None, *, half_image=False):
        del voxel_size
        if half_image:
            pixel_count = int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)
        else:
            pixel_count = int(image_shape[0]) * int(image_shape[1])
        return jnp.ones((params.shape[0], pixel_count), dtype=jnp.float32)

    @staticmethod
    def process_images(batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            16,
            axis=1,
        )

    @staticmethod
    def process_images_half(batch, apply_image_mask=False, **kwargs):
        del apply_image_mask, kwargs
        batch = jnp.asarray(batch)
        return jnp.repeat(
            batch[:, :1, :1].reshape(batch.shape[0], 1).astype(jnp.complex64),
            12,
            axis=1,
        )

    @property
    def image_mask(self):
        return np.ones(self.image_shape, dtype=np.float32)

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_units)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, indices.size, int(batch_size)):
            selected = indices[start : start + int(batch_size)]
            yield (
                jnp.asarray(self._images[selected]),
                self.rotation_matrices[selected],
                self.translations[selected],
                jnp.asarray(self.CTF_params[selected]),
                None,
                selected,
                selected,
            )

    def original_image_indices_from_local(self, indices):
        return np.asarray(indices, dtype=np.int64)


def test_coarse_gaussian_gemm_live_significance_contract_full_and_poisoned_tails(
    monkeypatch,
    tmp_path,
):
    """Live K-class pass preserves layout, priors, support, and both tail masks."""

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers import projection as projection_helpers
    from recovar.em.dense_single_volume.helpers import sparse_pass2_bucketed

    for name, value in {
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MACRO": "1",
        "RECOVAR_COARSE_GAUSSIAN_GEMM_MAX_PROJECTED_TRANSIENT_GB": "0.01",
        "RECOVAR_K1_COARSE_GAUSSIAN_FFI": "1",
        "RECOVAR_K1_COARSE_GAUSSIAN_SINCOSF": "1",
        "RECOVAR_K1_COARSE_FUSED_PROJECTOR": "0",
        "RECOVAR_RELION_COARSE_CANONICAL_REDUCTION": "0",
        "RECOVAR_K1_COARSE_NATIVE_ATOMIC_REDUCTION": "0",
        "RECOVAR_K1_COARSE_SINGLE_LANE_CANONICAL": "0",
        "RECOVAR_K1_COARSE_MULTISTREAM_WORKERS": "0",
        "RECOVAR_K1_COARSE_GAUSSIAN_NATIVE_TEXTURE": "0",
        "RECOVAR_K1_RELION_EXACT_COARSE_OPERANDS": "1",
        "RECOVAR_K1_RELION_F32_COARSE_SUPPORT": "0",
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(significance.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(cuda_backproject, "cuda_available", lambda: True)
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_exact_ctf_half_from_source_star",
        lambda _dataset, indices, image_shape: jnp.ones(
            (len(indices), int(image_shape[0]) * (int(image_shape[1]) // 2 + 1)),
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(
        sparse_pass2_bucketed,
        "_relion_cuda_powerclass_highres_xi2_half",
        lambda processed, **_kwargs: jnp.zeros(processed.shape[0], dtype=jnp.float32),
    )
    monkeypatch.setattr(
        cuda_backproject,
        "relion_translate_score_f32",
        lambda images, translation_angles, pixel_indices, image_shape: jnp.repeat(
            images[:, None, :],
            int(translation_angles.shape[0]),
            axis=1,
        ).reshape(images.shape[0] * int(translation_angles.shape[0]), -1),
    )

    projection_calls = []

    def fake_projection(projector_half, rotations_block, image_shape, **kwargs):
        del image_shape
        assert isinstance(kwargs["pixel_indices"], np.ndarray)
        class_index = int(np.rint(np.asarray(projector_half)[0, 0, 0].real))
        rotations_np = np.asarray(rotations_block)
        rotation_markers = rotations_np[:, 0, 1]
        is_poisoned_tail = rotation_markers == 0.0
        rotation_ids = rotation_markers - 10.0
        codes = class_index * 100.0 + rotation_ids * 10.0
        codes = np.where(is_poisoned_tail, 9999.0, codes).astype(np.float32)
        projection_calls.append((class_index, codes.copy(), is_poisoned_tail.copy()))
        projected = jnp.repeat(
            jnp.asarray(codes, dtype=jnp.complex64)[:, None],
            len(kwargs["pixel_indices"]),
            axis=1,
        )
        return projected, jnp.abs(projected) ** 2

    monkeypatch.setattr(
        projection_helpers,
        "compute_relion_projector_projections_block",
        fake_projection,
    )

    score_calls = []

    def designed_scores(projected, shifted, actual_count):
        active = jnp.arange(shifted.shape[0]) < actual_count
        safe_shifted = jnp.where(active[:, None, None], shifted, 0.0)
        image_ids = jnp.rint(jnp.max(safe_shifted.real, axis=(1, 2))).astype(jnp.int32) - 1
        desired = jnp.asarray([0.0, 121.0, 10.0], dtype=jnp.float32)[
            jnp.clip(image_ids, 0, 2)
        ]
        candidate = projected[:, 0].real[None, :, None] + jnp.arange(
            shifted.shape[1],
            dtype=jnp.float32,
        )[None, None, :]
        scores = -10.0 * jnp.abs(candidate - desired[:, None, None])
        return jnp.where(active[:, None, None], scores, 0.0)

    def controlled_scores(
        projected,
        projected_abs2,
        shifted,
        weight,
        initial,
        actual_image_count,
        **_kwargs,
    ):
        del projected_abs2, weight, initial
        shifted_np = np.asarray(shifted)
        actual_count = int(np.asarray(actual_image_count))
        if poison_tail["enabled"] and actual_count < shifted_np.shape[0]:
            assert np.isnan(shifted_np[actual_count:]).all()
        scores = designed_scores(projected, shifted, actual_count)
        score_calls.append((actual_count, tuple(shifted.shape)))
        return scores

    monkeypatch.setattr(
        significance,
        "_relion_coarse_gaussian_gemm_scores",
        controlled_scores,
    )

    original_pad = significance._pad_significance_preprocess_inputs
    poison_tail = {"enabled": False}

    def maybe_poison_tail(*args, **kwargs):
        result = list(original_pad(*args, **kwargs))
        if poison_tail["enabled"] and np.asarray(result[0]).shape[0] > np.asarray(args[0]).shape[0]:
            batch = np.asarray(result[0]).copy()
            ctf = np.asarray(result[1]).copy()
            batch[-1] = np.nan
            ctf[-1] = np.nan
            result[0] = batch
            result[1] = ctf
        return tuple(result)

    monkeypatch.setattr(
        significance,
        "_pad_significance_preprocess_inputs",
        maybe_poison_tail,
    )

    dataset = _MacroIntegrationDataset()
    rotations = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
    rotations[:, 0, 1] = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    translations = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    relion_projector = jnp.stack(
        [
            jnp.full((3, 3, 2), class_index, dtype=jnp.complex64)
            for class_index in range(2)
        ]
    )
    common = dict(
        class_log_priors=np.asarray([-0.1, -0.2], dtype=np.float64),
        adaptive_fraction=0.5,
        max_significants=1,
        image_batch_size=2,
        rotation_block_size=2,
        current_size=4,
        rotation_log_prior=np.asarray(
            [[0.0, -0.01, -0.02], [-0.03, -0.04, -0.05]],
            dtype=np.float32,
        ),
        translation_log_prior=np.asarray(
            [[0.0, -0.01], [-0.02, 0.0], [0.0, -0.03]],
            dtype=np.float32,
        ),
        half_spectrum_scoring=True,
        relion_projector_half=relion_projector,
        relion_projector_r_max=1,
        relion_projector_texture_interp=True,
        score_mode="gaussian",
        collect_significance=True,
        return_class_best=True,
        pad_final_image_batch=True,
    )

    clean = significance._compute_k_class_significance_batched(
        dataset,
        jnp.zeros((2, dataset.volume_size), dtype=jnp.complex64),
        jnp.ones(dataset.image_size, dtype=jnp.float32),
        rotations,
        translations,
        "linear_interp",
        **common,
    )
    diagnostic_dir = tmp_path / "paired_scores"
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_DIR",
        str(diagnostic_dir),
    )
    monkeypatch.setenv(
        "RECOVAR_COARSE_GAUSSIAN_GEMM_DIAGNOSTIC_ORIGINAL_INDICES",
        "0,1,2",
    )

    def direct_square_stub(projected, shifted, weight, initial, full_to_compact):
        del initial, full_to_compact
        active = np.any(np.asarray(weight) != 0.0, axis=1)
        actual_count = int(np.count_nonzero(active))
        return -designed_scores(projected, shifted, actual_count)

    monkeypatch.setattr(
        cuda_backproject,
        "relion_coarse_diff2_rectangular_f32",
        direct_square_stub,
    )
    poison_tail["enabled"] = True
    poisoned = significance._compute_k_class_significance_batched(
        dataset,
        jnp.zeros((2, dataset.volume_size), dtype=jnp.complex64),
        jnp.ones(dataset.image_size, dtype=jnp.float32),
        rotations,
        translations,
        "linear_interp",
        **common,
    )

    for clean_value, poisoned_value in zip(clean[:4], poisoned[:4]):
        np.testing.assert_array_equal(np.asarray(poisoned_value), np.asarray(clean_value))
    np.testing.assert_array_equal(clean[1], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(clean[2], np.asarray([0, 5, 2], dtype=np.int32))
    np.testing.assert_array_equal(clean[3], np.asarray([0, 1, 0], dtype=np.int32))
    expected_support = [
        [[0], [], [2]],
        [[], [5], []],
    ]
    for class_index in range(2):
        for image_index in range(3):
            np.testing.assert_array_equal(
                significance.significant_sample_ids(
                    clean[4][class_index][image_index],
                    6,
                ),
                np.asarray(expected_support[class_index][image_index], dtype=np.int64),
            )
    assert len(score_calls) == 16
    assert [call[0] for call in score_calls].count(2) == 8
    assert [call[0] for call in score_calls].count(1) == 8
    assert any(np.any(tail) for _, _, tail in projection_calls)
    resources = clean[5]["coarse_gaussian_gemm_resources"]
    assert resources["pixel_index_device_to_host_materializations"] == 0
    assert resources["predicted_peak_projection_bytes"] <= resources[
        "projected_transient_budget_bytes"
    ]
    clean_qualification = clean[5]["coarse_gaussian_gemm_qualification"]
    assert clean_qualification["paired_capture_active"] is False
    assert clean_qualification["clean_timing_eligible"] is True
    captured_qualification = poisoned[5]["coarse_gaussian_gemm_qualification"]
    assert captured_qualification["paired_capture_active"] is True
    assert captured_qualification["clean_timing_eligible"] is False
    assert "separate diagnostic-off timing arm" in captured_qualification["timing_policy"]
    diagnostic_paths = sorted(diagnostic_dir.glob("*.npz"))
    assert len(diagnostic_paths) == 2
    captured_original_indices = []
    for diagnostic_path in diagnostic_paths:
        with np.load(diagnostic_path) as payload:
            assert str(payload["layout"]) == "image,class,rotation,translation"
            assert bool(payload["paired_capture_active"]) is True
            assert bool(payload["clean_timing_eligible"]) is False
            assert "separate_diagnostic-off_timing_arm" in str(payload["timing_policy"])
            np.testing.assert_array_equal(
                payload["direct_scores_with_prior"],
                payload["macro_scores_with_prior"],
            )
            np.testing.assert_array_equal(payload["argmax_equal"], True)
            np.testing.assert_array_equal(payload["support_equal"], True)
            assert int(payload["resource_pixel_index_device_to_host_materializations"]) == 0
            captured_original_indices.extend(payload["original_indices"].tolist())
    assert sorted(captured_original_indices) == [0, 1, 2]


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
