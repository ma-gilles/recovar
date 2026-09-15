from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.scoring import scoring
from recovar.em.scoring.coarse_gemm_hybrid import (
    _upward_add_f64,
    _upward_multiply_f64,
    certified_f64_expanded_score_gammas,
    coarse_gemm_direct_f32_ftz_envelope_and_range,
    coarse_gemm_direct_score_intervals,
    coarse_gemm_expanded_score_eta_f64,
    initialize_coarse_gemm_hybrid_interval_state,
    plan_coarse_gemm_certificate_topology,
    propagate_coarse_gemm_intervals_through_f32_priors,
    update_coarse_gemm_hybrid_interval_state,
)

pytestmark = pytest.mark.unit


def _upward_gamma(operation_count: int, unit_roundoff: float) -> float:
    scaled = np.float64(operation_count) * np.float64(unit_roundoff)
    denominator = np.nextafter(np.float64(1.0) - scaled, np.float64(-np.inf))
    return float(np.nextafter(scaled / denominator, np.float64(np.inf)))


def test_promoted_expanded_score_gammas_use_reviewed_gf46_counts() -> None:
    bounds = certified_f64_expanded_score_gammas(5100)

    assert bounds.cross == _upward_gamma(10_204, 2.0**-53)
    assert bounds.energy == _upward_gamma(5_105, 2.0**-53)
    assert bounds.initial_diff2 == _upward_gamma(3, 2.0**-53)
    assert bounds.energy_envelope == _upward_gamma(5_102, 2.0**-53)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_promoted_expanded_score_gammas_reject_invalid_bounds(value) -> None:
    with pytest.raises(ValueError):
        certified_f64_expanded_score_gammas(value)


def test_certificate_topology_is_bound_to_the_actual_direct_lookup() -> None:
    mapping = np.asarray([0, -1, 1, 2, -1, 3], dtype=np.int32)
    topology = plan_coarse_gemm_certificate_topology(
        mapping,
        compact_pixel_count=4,
        translation_count=3,
    )

    assert topology.full_position_count == 6
    assert topology.compact_pixel_count == 4
    assert topology.translation_count == 3
    assert len(topology.full_to_compact_sha256) == 64
    assert topology.expanded_f64_gammas == certified_f64_expanded_score_gammas(6)


@pytest.mark.parametrize(
    "mapping",
    [
        np.asarray([0, 1, 1, 3], dtype=np.int32),
        np.asarray([0, 1, 2, 4], dtype=np.int32),
        np.asarray([0, 1, 2, 3], dtype=np.int64),
    ],
)
def test_certificate_topology_rejects_missing_duplicate_or_wrong_dtype_mapping(mapping) -> None:
    with pytest.raises(ValueError):
        plan_coarse_gemm_certificate_topology(
            mapping,
            compact_pixel_count=4,
            translation_count=3,
        )


def test_promoted_eta_matches_each_directed_envelope_operation() -> None:
    reference = np.asarray([[4.0, 9.0]], dtype=np.float64)
    image = np.asarray([[16.0, 25.0, 36.0]], dtype=np.float64)
    initial = np.asarray([0.5], dtype=np.float64)
    coefficients = certified_f64_expanded_score_gammas(32)
    with jax.enable_x64(True):
        actual = np.asarray(
            coarse_gemm_expanded_score_eta_f64(
                reference,
                image,
                initial,
                cross_gamma=coefficients.cross,
                energy_gamma=coefficients.energy,
                initial_diff2_gamma=coefficients.initial_diff2,
                energy_envelope_gamma=coefficients.energy_envelope,
            )
        )

    positive_inf = np.float64(np.inf)
    negative_inf = np.float64(-np.inf)
    denominator = np.nextafter(
        np.float64(1.0) - coefficients.energy_envelope,
        negative_inf,
    )
    reference_upper = np.nextafter(reference / denominator, positive_inf)[:, :, None]
    image_upper = np.nextafter(image / denominator, positive_inf)[:, None, :]
    energy_sum = np.nextafter(reference_upper + image_upper, positive_inf)
    product = np.nextafter(reference_upper * image_upper, positive_inf)
    cross_envelope = np.nextafter(np.sqrt(product), positive_inf)
    cross_term = np.nextafter(coefficients.cross * cross_envelope, positive_inf)
    energy_term = np.nextafter(coefficients.energy * energy_sum, positive_inf)
    energy_term = np.nextafter(np.float64(0.5) * energy_term, positive_inf)
    initial_term = np.nextafter(
        coefficients.initial_diff2 * initial[:, None, None],
        positive_inf,
    )
    expected = np.nextafter(
        np.nextafter(cross_term + energy_term, positive_inf) + initial_term,
        positive_inf,
    )

    assert np.array_equal(actual, expected)


def test_direct_f32_ftz_envelope_is_compact_and_negligible_at_ordinary_scale() -> None:
    rho, valid = coarse_gemm_direct_f32_ftz_envelope_and_range(
        np.asarray([1.0, 4.0], dtype=np.float64),
        np.asarray([2.0, 8.0, 16.0], dtype=np.float64),
        np.asarray([0.5, 1.0, 2.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.int32(5100),
        np.float64(1.0e-4),
    )

    rho = np.asarray(rho)
    valid = np.asarray(valid)
    assert rho.shape == valid.shape == (3, 2, 1)
    assert np.all(np.isfinite(rho) & (rho > 0.0))
    assert np.all(rho < 1.0e-20)
    assert np.all(valid)


def test_nested_range_products_are_individually_outward_rounded() -> None:
    # Regression for a chain where one nextafter after two multiplications is
    # still below the exact binary64-input product.
    reference = np.float64(np.float32(238_228.88))
    image = np.float64(np.float32(2_309_324_800_000.0))
    weight = np.float64(np.float32(3_941_990_700_000.0))
    difference = np.asarray(_upward_add_f64(reference, image)).item()
    square = np.asarray(_upward_multiply_f64(difference, difference)).item()
    product = np.asarray(_upward_multiply_f64(square, weight)).item()

    exact = _fraction(difference) * _fraction(difference) * _fraction(weight)
    assert _fraction(square) >= _fraction(difference) * _fraction(difference)
    assert _fraction(product) >= exact


def test_direct_f32_range_gate_rejects_local_square_overflow_even_at_zero_weight() -> None:
    rho, valid = coarse_gemm_direct_f32_ftz_envelope_and_range(
        np.asarray([1.0e20], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.int32(1),
        np.float64(1.0e-6),
    )

    assert np.asarray(rho).shape == (1, 1, 1)
    assert not np.asarray(valid).item()


def test_direct_interval_rejects_false_range_gate_and_adds_ftz_separately() -> None:
    score = np.asarray([[[-1.0]]], dtype=np.float64)
    eta = np.float64(0.0)
    gamma = np.float64(0.0)
    rho = np.float64(2.0**-40)
    lower, upper = coarse_gemm_direct_score_intervals(
        score,
        eta,
        gamma,
        rho,
        np.asarray([[[True]]], dtype=bool),
    )
    expected_error = np.nextafter(rho, np.float64(np.inf))
    expected_error = np.nextafter(expected_error, np.float64(np.inf))
    assert np.array_equal(
        np.asarray(lower),
        np.nextafter(score - expected_error, np.float64(-np.inf)),
    )
    assert np.array_equal(
        np.asarray(upper),
        np.nextafter(score + expected_error, np.float64(np.inf)),
    )

    invalid = coarse_gemm_direct_score_intervals(
        score,
        eta,
        gamma,
        rho,
        np.asarray([[[False]]], dtype=bool),
    )
    assert all(np.isnan(np.asarray(endpoint)).all() for endpoint in invalid)


def _stored_operands():
    projected = np.asarray(
        [
            [1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 0.5j, 3.0 + 0.0j],
            [-0.5 + 2.0j, 1.5 - 1.0j, 0.0 + 0.25j, -1.0 - 2.0j],
        ],
        dtype=np.complex64,
    )
    shifted = np.asarray(
        [
            [
                [0.5 + 0.25j, -1.0 + 1.5j, 0.5 - 0.25j, 2.0 + 0.5j],
                [1.5 - 0.5j, -2.5 + 0.0j, 0.0 + 0.5j, 3.5 - 0.5j],
                [0.0 + 0.0j, 1.0 + 1.0j, -0.5 - 0.5j, 1.0 + 2.0j],
            ],
            [
                [1.0 - 1.0j, 0.5 + 0.5j, 0.25 + 0.25j, -2.0 + 0.0j],
                [2.0 + 1.0j, -1.0 - 1.0j, 0.0 - 0.5j, 1.0 + 1.0j],
                [-1.0 + 0.5j, 1.5 + 0.0j, 0.5 + 0.0j, 2.0 - 1.0j],
            ],
        ],
        dtype=np.complex64,
    )
    weight = np.asarray(
        [[1.0, 0.5, 2.0, 0.25], [0.25, 1.5, 0.5, 2.0]],
        dtype=np.float32,
    )
    initial = np.asarray([0.5, 1.25], dtype=np.float32)
    return projected, shifted, weight, initial


def _stored_topology(*, n_pixels: int = 4, n_translations: int = 3):
    return plan_coarse_gemm_certificate_topology(
        np.arange(n_pixels, dtype=np.int32),
        compact_pixel_count=n_pixels,
        translation_count=n_translations,
    )


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_stored_score(projected, shifted, weight, initial, image, rotation, translation):
    residual = _fraction(initial[image])
    for pixel in range(projected.shape[1]):
        real_delta = _fraction(projected[rotation, pixel].real) - _fraction(
            shifted[image, translation, pixel].real
        )
        imag_delta = _fraction(projected[rotation, pixel].imag) - _fraction(
            shifted[image, translation, pixel].imag
        )
        residual += Fraction(1, 2) * _fraction(weight[image, pixel]) * (
            real_delta * real_delta + imag_delta * imag_delta
        )
    return -residual


@pytest.mark.parametrize("real_cross", [False, True])
def test_promoted_certificate_encloses_exact_stored_residual_candidatewise(real_cross) -> None:
    projected, shifted, weight, initial = _stored_operands()
    with jax.enable_x64(False):
        assert not jax.config.x64_enabled
        result = scoring._relion_coarse_gaussian_gemm_certificate(
            projected,
            shifted,
            weight,
            initial,
            2,
            topology=_stored_topology(),
            real_cross=real_cross,
        )
        assert not jax.config.x64_enabled

    lower = np.asarray(result.raw_lower)
    upper = np.asarray(result.raw_upper)
    assert result.macro_scores.dtype == jnp.float32
    assert result.raw_lower.dtype == result.raw_upper.dtype == jnp.float64
    assert np.all(np.isfinite(lower))
    assert np.all(lower <= upper)
    for image in range(shifted.shape[0]):
        for rotation in range(projected.shape[0]):
            for translation in range(shifted.shape[1]):
                exact = _exact_stored_score(
                    projected,
                    shifted,
                    weight,
                    initial,
                    image,
                    rotation,
                    translation,
                )
                assert _fraction(lower[image, rotation, translation]) <= exact
                assert exact <= _fraction(upper[image, rotation, translation])


@pytest.mark.parametrize("real_cross", [False, True])
def test_certificate_fails_closed_before_direct_f32_square_overflow(real_cross) -> None:
    projected = np.asarray([[np.complex64(1.0e20 + 0.0j)]], dtype=np.complex64)
    shifted = np.zeros((1, 1, 1), dtype=np.complex64)
    weight = np.ones((1, 1), dtype=np.float32)
    initial = np.zeros((1,), dtype=np.float32)

    result = scoring._relion_coarse_gaussian_gemm_certificate(
        projected,
        shifted,
        weight,
        initial,
        1,
        topology=_stored_topology(n_pixels=1, n_translations=1),
    )

    assert np.isneginf(np.asarray(result.macro_scores)).all()
    assert np.isnan(np.asarray(result.raw_lower)).all()
    assert np.isnan(np.asarray(result.raw_upper)).all()


@pytest.mark.parametrize("real_cross", [False, True])
def test_certificate_ftz_envelope_contains_underflowed_direct_f32_score(real_cross) -> None:
    projected = np.asarray([[np.complex64(1.0e-30 + 0.0j)]], dtype=np.complex64)
    shifted = np.zeros((1, 1, 1), dtype=np.complex64)
    weight = np.ones((1, 1), dtype=np.float32)
    initial = np.zeros((1,), dtype=np.float32)

    result = scoring._relion_coarse_gaussian_gemm_certificate(
        projected,
        shifted,
        weight,
        initial,
        1,
        topology=_stored_topology(n_pixels=1, n_translations=1),
    )

    # The source16 operation sequence squares in FP32, so this stored input
    # contributes zero with or without FTZ.  The promoted mathematical center
    # is nonzero and therefore needs the separate additive envelope.
    direct_score = np.float32(-0.5) * np.float32(
        np.float32(projected[0, 0].real) * np.float32(projected[0, 0].real)
    )
    lower = np.asarray(result.raw_lower)[0, 0, 0]
    upper = np.asarray(result.raw_upper)[0, 0, 0]
    assert direct_score == np.float32(0.0)
    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= np.float64(direct_score) <= upper


@pytest.mark.parametrize("real_cross", [False, True])
def test_certificate_masks_padded_garbage_before_promoted_arithmetic(real_cross) -> None:
    projected, shifted, weight, initial = _stored_operands()
    shifted[1] = np.complex64(np.nan + 1j * np.nan)
    weight[1] = np.float32(np.nan)
    initial[1] = np.float32(-1.0)

    result = scoring._relion_coarse_gaussian_gemm_certificate(
        projected,
        shifted,
        weight,
        initial,
        1,
        topology=_stored_topology(),
        real_cross=real_cross,
    )

    assert np.all(np.isfinite(np.asarray(result.raw_lower[0])))
    assert np.array_equal(np.asarray(result.macro_scores[1]), np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(np.asarray(result.raw_lower[1]), np.zeros((2, 3), dtype=np.float64))
    assert np.array_equal(np.asarray(result.raw_upper[1]), np.zeros((2, 3), dtype=np.float64))


@pytest.mark.parametrize("invalid", ["weight", "initial", "reference", "image"])
@pytest.mark.parametrize("real_cross", [False, True])
def test_certificate_turns_invalid_active_stored_inputs_into_nan_endpoints(invalid, real_cross) -> None:
    projected, shifted, weight, initial = _stored_operands()
    if invalid == "weight":
        weight[0, 0] = -1.0
    elif invalid == "initial":
        initial[0] = -1.0
    elif invalid == "reference":
        projected[0, 0] = np.complex64(np.nan + 0.0j)
    else:
        shifted[0, 0, 0] = np.complex64(np.inf + 0.0j)

    result = scoring._relion_coarse_gaussian_gemm_certificate(
        projected,
        shifted,
        weight,
        initial,
        2,
        topology=_stored_topology(),
        real_cross=real_cross,
    )

    if invalid == "reference":
        assert np.isnan(np.asarray(result.raw_lower)).all()
    else:
        assert np.isnan(np.asarray(result.raw_lower[0])).all()


def test_certificate_rejects_wrong_storage_dtype_shape_and_topology() -> None:
    projected, shifted, weight, initial = _stored_operands()
    with pytest.raises(TypeError, match="dtype is invalid"):
        scoring._relion_coarse_gaussian_gemm_certificate(
            projected.astype(np.complex128),
            shifted,
            weight,
            initial,
            2,
            topology=_stored_topology(),
        )
    with pytest.raises(ValueError, match="inconsistent shapes"):
        scoring._relion_coarse_gaussian_gemm_certificate(
            projected,
            shifted,
            weight[:, :-1],
            initial,
            2,
            topology=_stored_topology(),
        )
    with pytest.raises(ValueError, match="topology does not match"):
        scoring._relion_coarse_gaussian_gemm_certificate(
            projected,
            shifted,
            weight,
            initial,
            2,
            topology=_stored_topology(n_pixels=3),
        )


@pytest.mark.parametrize("real_cross", [False, True])
def test_fused_certificate_state_update_matches_explicit_interval_pipeline(real_cross) -> None:
    projected_seed, shifted, weight, initial = _stored_operands()
    projected = np.tile(projected_seed, (8, 1))
    rotation_prior = np.linspace(-2.0, 1.0, 16, dtype=np.float32)
    translation_prior = np.asarray(
        [[0.25, -0.5, 1.0], [-1.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    class_prior = np.float32(-0.75)

    explicit = initialize_coarse_gemm_hybrid_interval_state(2, 16)
    certificate = scoring._relion_coarse_gaussian_gemm_certificate(
        projected,
        shifted,
        weight,
        initial,
        2,
        topology=_stored_topology(),
        real_cross=real_cross,
    )
    posterior_lower, posterior_upper = propagate_coarse_gemm_intervals_through_f32_priors(
        certificate.raw_lower,
        certificate.raw_upper,
        class_log_prior=class_prior,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )
    explicit = update_coarse_gemm_hybrid_interval_state(
        explicit,
        certificate.raw_lower,
        certificate.raw_upper,
        posterior_lower,
        posterior_upper,
        rotation_offset=0,
        valid_rotation_count=16,
        actual_image_count=2,
    )

    image_batch = scoring._prepare_relion_coarse_gaussian_gemm_f64_image_batch(
        shifted,
        weight,
        initial,
        2,
    )
    fused = scoring._relion_coarse_gaussian_gemm_update_certificate_state(
        initialize_coarse_gemm_hybrid_interval_state(2, 16),
        projected,
        image_batch,
        topology=_stored_topology(),
        real_cross=real_cross,
        rotation_offset=0,
        class_log_prior=class_prior,
        rotation_log_prior=rotation_prior,
        translation_log_prior=translation_prior,
    )

    assert all(
        np.array_equal(np.asarray(fused_value), np.asarray(explicit_value))
        for fused_value, explicit_value in zip(fused, explicit)
    )
    assert [tuple(value.shape) for value in jax.tree.leaves(fused)] == [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (16,),
        (2,),
        (2,),
    ]


def test_fused_certificate_state_update_rejects_unaligned_source_blocks() -> None:
    projected_seed, shifted, weight, initial = _stored_operands()
    projected = np.tile(projected_seed, (8, 1))
    state = initialize_coarse_gemm_hybrid_interval_state(2, 32)
    image_batch = scoring._prepare_relion_coarse_gaussian_gemm_f64_image_batch(
        shifted,
        weight,
        initial,
        2,
    )

    with pytest.raises(ValueError, match="aligned complete source-16 blocks"):
        scoring._relion_coarse_gaussian_gemm_update_certificate_state(
            state,
            projected,
            image_batch,
            topology=_stored_topology(),
            rotation_offset=1,
            class_log_prior=np.float32(0.0),
        )


def test_fused_certificate_lowering_keeps_full_endpoints_transient() -> None:
    topology = _stored_topology()
    with jax.enable_x64(True):
        state = initialize_coarse_gemm_hybrid_interval_state(2, 16)
        projected = jnp.ones((16, 4), dtype=jnp.complex64)
        shifted = jnp.ones((2, 3, 4), dtype=jnp.complex64)
        weight = jnp.ones((2, 4), dtype=jnp.float32)
        initial = jnp.zeros((2,), dtype=jnp.float32)
        image_batch = scoring._prepare_relion_coarse_gaussian_gemm_f64_image_batch(
            shifted,
            weight,
            initial,
            2,
        )
        lowered = scoring._relion_coarse_gaussian_gemm_update_certificate_state_jit.lower(
            state,
            projected,
            image_batch,
            jnp.float32(0.0),
            None,
            None,
            *topology.expanded_f64_gammas,
            # This focused structural check does not depend on the coefficient
            # value; production obtains it from the reviewed topology helper.
            np.float64(1e-6),
            np.int32(topology.full_position_count),
            jnp.int32(0),
        )
        stablehlo = str(lowered.compiler_ir("stablehlo"))
        memory = lowered.compile().memory_analysis()

    assert stablehlo.count("stablehlo.dot_general") == 2
    # The only abs operations are the two real-component maxima used by the
    # provisional FP32 range/FTZ guard.  Energy remains explicit real^2+imag^2
    # and is still shared with the two mature GEMM components.
    assert stablehlo.count("stablehlo.abs") == 2
    assert all(
        "tensor<16x4xf64>" in line
        for line in stablehlo.splitlines()
        if "stablehlo.abs" in line
    )
    # The public result is only the seven compact state buffers, not three
    # [image, rotation, translation] score/endpoint cubes.
    assert memory.output_size_in_bytes < 2 * 2 * 16 * 3 * np.dtype(np.float64).itemsize


def test_existing_windowed_score_uses_the_factored_mature_gemm_components() -> None:
    rng = np.random.default_rng(13330442)
    n_images, n_trans, n_rotations, n_pixels = 2, 3, 4, 7
    shifted = rng.normal(size=(n_images * n_trans, n_pixels)).astype(np.float32)
    weight = rng.uniform(0.5, 1.5, size=(n_images, n_pixels)).astype(np.float32)
    projected = rng.normal(size=(n_rotations, n_pixels)).astype(np.float32)
    projected_abs2 = projected * projected

    actual = np.asarray(
        scoring._e_step_block_scores_windowed(
            shifted,
            np.zeros((n_images, 1), dtype=np.float32),
            weight,
            projected,
            projected_abs2,
            np.ones(n_pixels, dtype=np.float32),
            n_images,
            n_trans,
            n_pixels,
            (8, 8),
            (8, 8, 8),
        )
    )
    cross = -2.0 * jnp.matmul(
        jnp.asarray(shifted),
        jnp.asarray(projected).T,
        precision=jax.lax.Precision.HIGHEST,
    )
    cross = cross.reshape(n_images, n_trans, n_rotations).swapaxes(1, 2)
    norms = jnp.matmul(
        jnp.asarray(weight),
        jnp.asarray(projected_abs2).T,
        precision=jax.lax.Precision.HIGHEST,
    )
    expected = np.asarray(-0.5 * (cross + norms[..., None]))

    assert np.array_equal(actual, expected)
