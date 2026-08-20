"""Tests for the pure-NumPy normalized-CC reduction replay diagnostic."""

import numpy as np
import pytest

from recovar.em.dense_single_volume.helpers.significance import (
    _relion_cc_inverse_power_from_processed,
)
from recovar.em.normalized_cc_replay import (
    RELION_COARSE_REDUCTION_LANES,
    RELION_FINE_REDUCTION_LANES,
    REPLAY_SCHEMA,
    canonical_float32_reduce,
    canonical_float64_reduce,
    classify_normalized_cc_candidate_replays,
    normalized_cc_pixel_contributions,
    recovar_logical_float32_reduce,
    relion_128lane_float32_reduce,
    relion_256lane_float32_reduce,
    replay_normalized_cc,
    replay_normalized_cc_candidates,
)

pytestmark = pytest.mark.unit


def _production_inputs():
    rng = np.random.default_rng(161552)
    batch, n_rot, n_trans, n_pix = 1, 3, 4, 17
    shifted = (rng.normal(size=(batch, n_trans, n_pix)) + 1j * rng.normal(size=(batch, n_trans, n_pix))).astype(
        np.complex64
    )
    projections = (rng.normal(size=(batch, n_rot, n_pix)) + 1j * rng.normal(size=(batch, n_rot, n_pix))).astype(
        np.complex64
    )
    score_weight = rng.uniform(0.1, 1.5, size=(batch, n_pix)).astype(np.float32)
    half_weight = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=n_pix)
    return shifted, score_weight, projections, half_weight


def _relion_coarse_atomic_score(numerator, norm):
    """Independent replay of all 128 identical coarse-kernel atomic adds."""

    contribution = np.float32(
        np.float32(numerator)
        / (np.float32(128.0) * np.sqrt(np.float32(norm)))
    )
    accumulated = np.float32(0.0)
    for _ in range(128):
        accumulated = np.float32(accumulated + contribution)
    return accumulated


def test_recovar_logical_replay_matches_production_normalized_cc_score():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.sparse_pass2_bucketed import (
        _score_pass2_pairs_normalized_cc,
    )

    shifted, score_weight, projections, half_weight = _production_inputs()
    production = np.asarray(
        _score_pass2_pairs_normalized_cc(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weight),
            jnp.asarray([[0]], dtype=jnp.int32),
            jnp.asarray([[0]], dtype=jnp.int32),
            jnp.asarray([[True]]),
        )
    )[0, 0]
    contributions = normalized_cc_pixel_contributions(
        projections[0, 0],
        shifted[0, 0],
        score_weight[0],
        half_weight,
    )
    replay = replay_normalized_cc(contributions)

    replay_bits = np.asarray(replay.recovar_logical_float32.score, dtype=np.float32).view(np.uint32)
    # XLA is free to select a backend-specific reduction tree; the pure NumPy
    # flat fold is a logical control, not a bitwise device replay.  Device-
    # captured contributions plus the explicit RELION reducers below provide
    # the bitwise replay path.
    production_bits = production.view(np.uint32)
    assert abs(int(replay_bits) - int(production_bits)) <= 1
    assert replay.schema == REPLAY_SCHEMA
    assert replay.schema_version == 1


def test_relion_256lane_reduction_matches_hand_reference():
    rng = np.random.default_rng(256)
    values = rng.normal(size=777).astype(np.float32)

    # Independent transcription of diff2.cuh: each lane walks its passes,
    # followed by the explicit 128, 64, ..., 1 shared-memory tree.
    lanes = np.zeros(RELION_FINE_REDUCTION_LANES, dtype=np.float32)
    for lane in range(RELION_FINE_REDUCTION_LANES):
        for pixel in range(lane, values.size, RELION_FINE_REDUCTION_LANES):
            lanes[lane] = np.float32(lanes[lane] + values[pixel])
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        for lane in range(stride):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + stride])

    actual = relion_256lane_float32_reduce(values)
    assert actual.view(np.uint32) == lanes[0].view(np.uint32)


def test_relion_128lane_coarse_reduction_matches_hand_reference():
    rng = np.random.default_rng(128)
    values = rng.normal(size=649).astype(np.float32)

    # Independent transcription of cuda_kernel_diff2_CC_coarse: each lane
    # walks its pixel passes, followed by the explicit shared-memory tree.
    lanes = np.zeros(RELION_COARSE_REDUCTION_LANES, dtype=np.float32)
    for lane in range(RELION_COARSE_REDUCTION_LANES):
        for pixel in range(lane, values.size, RELION_COARSE_REDUCTION_LANES):
            lanes[lane] = np.float32(lanes[lane] + values[pixel])
    for stride in (64, 32, 16, 8, 4, 2, 1):
        for lane in range(stride):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + stride])

    actual = relion_128lane_float32_reduce(values)
    assert actual.view(np.uint32) == lanes[0].view(np.uint32)


def test_relion_coarse_rescore_matches_numpy_replay():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from recovar.em.dense_single_volume.helpers.scoring import (
        _relion_coarse_128lane_float32_reduce,
        _relion_coarse_cc_atomic_score_from_components,
        _relion_coarse_normalized_cc_rescore_jax,
    )

    rng = np.random.default_rng(6322)
    n_candidates, n_pixels = 4, 56 * 29
    shifted = (
        rng.normal(size=(n_candidates, n_pixels))
        + 1j * rng.normal(size=(n_candidates, n_pixels))
    ).astype(np.complex64)
    projections = (
        rng.normal(size=(n_candidates, n_pixels))
        + 1j * rng.normal(size=(n_candidates, n_pixels))
    ).astype(np.complex64)
    score_weight = rng.uniform(0.1, 1.5, size=(n_candidates, n_pixels)).astype(np.float32)
    half_weights = rng.choice(np.asarray([1.0, 2.0], dtype=np.float32), size=n_pixels)
    fftw_order = rng.permutation(n_pixels).astype(np.int32)

    values = rng.normal(size=(3, n_pixels)).astype(np.float32)
    reduced = np.asarray(_relion_coarse_128lane_float32_reduce(jnp.asarray(values)))
    expected_reduced = np.asarray([relion_128lane_float32_reduce(row) for row in values])
    np.testing.assert_array_equal(reduced.view(np.uint32), expected_reduced.view(np.uint32))

    actual = np.asarray(
        _relion_coarse_normalized_cc_rescore_jax(
            jnp.asarray(shifted),
            jnp.asarray(score_weight),
            jnp.asarray(projections),
            jnp.asarray(half_weights),
            jnp.asarray(fftw_order),
        )
    )
    expected = []
    for candidate in range(n_candidates):
        contributions = normalized_cc_pixel_contributions(
            projections[candidate, fftw_order],
            shifted[candidate, fftw_order],
            score_weight[candidate, fftw_order],
            half_weights[fftw_order],
        )
        numerator = relion_128lane_float32_reduce(contributions.numerator)
        norm = relion_128lane_float32_reduce(contributions.norm)
        expected.append(_relion_coarse_atomic_score(numerator, norm))
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)

    # Frozen case-4 cross-winner components: direct division differs by up
    # to six ULPs, while RELION's 128 atomic additions collapse both scores
    # to the same bit pattern.
    frozen_numerator = jnp.asarray(
        [0.09698139131069183, 0.09905915707349777],
        dtype=jnp.float32,
    )
    frozen_norm = jnp.asarray(
        [0.12128135561943054, 0.12653392553329468],
        dtype=jnp.float32,
    )
    frozen_actual = np.asarray(
        _relion_coarse_cc_atomic_score_from_components(
            frozen_numerator,
            frozen_norm,
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        frozen_actual.view(np.uint32),
        np.asarray([1049531574, 1049531574], dtype=np.uint32),
    )


def test_relion_coarse_native_rescore_matches_exact_uniform_operands():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject
    from recovar.em.dense_single_volume.helpers.scoring import (
        _relion_coarse_normalized_cc_rescore,
        _relion_coarse_normalized_cc_rescore_jax,
    )

    if jax.default_backend() != "gpu" or not cuda_backproject.custom_cuda_requested():
        pytest.skip("native RELION coarse rescore requires the custom CUDA GPU path")

    n_pixels = 56 * 29
    shifted = jnp.ones((2, n_pixels), dtype=jnp.complex64)
    score_weight = jnp.ones((2, n_pixels), dtype=jnp.float32)
    projection = jnp.ones((2, n_pixels), dtype=jnp.complex64)
    half_weights = jnp.ones(n_pixels, dtype=jnp.float32)
    fftw_order = jnp.arange(n_pixels, dtype=jnp.int32)

    actual = np.asarray(
        _relion_coarse_normalized_cc_rescore(
            shifted,
            score_weight,
            projection,
            half_weights,
            fftw_order,
        ),
        dtype=np.float32,
    )
    expected = np.asarray(
        _relion_coarse_normalized_cc_rescore_jax(
            shifted,
            score_weight,
            projection,
            half_weights,
            fftw_order,
        ),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_relion_coarse_native_texture_rescore_exposes_reduced_components():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from recovar import cuda_backproject

    if jax.default_backend() != "gpu" or not cuda_backproject.custom_cuda_requested():
        pytest.skip("native RELION texture rescore requires the custom CUDA GPU path")

    projector = jnp.ones((5, 5, 5), dtype=jnp.complex64) * (1.0 + 1.0j)
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (2, 3, 3))
    shifted = jnp.ones((2, 4), dtype=jnp.complex64)
    score_weight = jnp.ones((2, 4), dtype=jnp.float32)
    half_weights = jnp.ones((4,), dtype=jnp.float32)
    packed_to_compact = jnp.arange(4, dtype=jnp.int32)

    components = np.asarray(
        cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32(
            projector,
            rotations,
            shifted,
            score_weight,
            half_weights,
            packed_to_compact,
            2,
            1,
            1,
            True,
        ),
        dtype=np.float32,
    )

    assert components.shape == (2, 3)
    np.testing.assert_array_equal(components[0].view(np.uint32), components[1].view(np.uint32))
    assert np.all(components[:, 1:] > 0.0)

    translated = np.asarray(
        cuda_backproject.relion_coarse_normalized_cc_native_texture_pairs_f32(
            projector,
            rotations,
            shifted,
            score_weight,
            half_weights,
            packed_to_compact,
            2,
            1,
            1,
            True,
            translation_angles=jnp.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=jnp.float32),
        ),
        dtype=np.float32,
    )
    assert translated[0, 0].view(np.uint32) != translated[1, 0].view(np.uint32)


def test_relion_coarse_exact_tie_uses_direction_major_flat_order():
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_pose_tie_break_keys,
        _select_relion_coarse_rescore_winner_slots,
    )

    candidate_pose_ids = np.asarray([[955081, 977030]], dtype=np.int64)
    keys = _relion_coarse_pose_tie_break_keys(
        candidate_pose_ids,
        n_trans=29,
        healpix_order=3,
    )
    np.testing.assert_array_equal(
        keys,
        np.asarray([[943626, 928339]], dtype=np.int64),
    )
    slots, tie_count = _select_relion_coarse_rescore_winner_slots(
        np.asarray([[0.27847832441329956, 0.27847832441329956]], dtype=np.float32),
        candidate_pose_ids,
        n_trans=29,
        healpix_order=3,
    )
    np.testing.assert_array_equal(slots, np.asarray([1], dtype=np.int32))
    assert tie_count == 1


def test_relion_coarse_tie_order_maps_subset_rotation_ids():
    from recovar.em.dense_single_volume.helpers.significance import (
        _relion_coarse_pose_tie_break_keys,
        _select_relion_coarse_rescore_winner_slots,
    )

    canonical_rotation_ids = np.asarray([32933, 33690], dtype=np.int64)
    local_candidate_pose_ids = np.asarray([[24, 49]], dtype=np.int64)
    keys = _relion_coarse_pose_tie_break_keys(
        local_candidate_pose_ids,
        n_trans=29,
        healpix_order=3,
        coarse_rotation_ids=canonical_rotation_ids,
    )
    np.testing.assert_array_equal(
        keys,
        np.asarray([[943626, 928339]], dtype=np.int64),
    )
    slots, tie_count = _select_relion_coarse_rescore_winner_slots(
        np.asarray([[0.27847832441329956, 0.27847832441329956]], dtype=np.float32),
        local_candidate_pose_ids,
        n_trans=29,
        healpix_order=3,
        coarse_rotation_ids=canonical_rotation_ids,
    )
    np.testing.assert_array_equal(slots, np.asarray([1], dtype=np.int32))
    assert tie_count == 1


def test_reduction_order_can_flip_near_tie_while_float64_agrees():
    # A's exact sum is 1e-7, but a flat float32 fold loses it between the two
    # 1e8 terms.  RELION's lane tree combines the cancelling terms before the
    # small term.  Candidate B sits between the two results.
    candidate_a = np.asarray([1e8, 1e-7, -1e8], dtype=np.float32)
    candidate_b = np.asarray([5e-8, 0.0, 0.0], dtype=np.float32)

    recovar_scores = [
        recovar_logical_float32_reduce(candidate_a),
        recovar_logical_float32_reduce(candidate_b),
    ]
    relion_scores = [
        relion_256lane_float32_reduce(candidate_a),
        relion_256lane_float32_reduce(candidate_b),
    ]
    canonical_f64_scores = [
        canonical_float64_reduce(candidate_a),
        canonical_float64_reduce(candidate_b),
    ]

    assert np.argmax(recovar_scores) == 1
    assert np.argmax(relion_scores) == 0
    assert np.argmax(canonical_f64_scores) == 0
    assert canonical_f64_scores[0] > canonical_f64_scores[1]


def test_canonical_reductions_use_common_integer_identity_order():
    values = np.asarray([3.0, 1e8, -1e8], dtype=np.float32)
    identities = np.asarray([2, 0, 1], dtype=np.int64)
    canonical_values = values[np.argsort(identities)]

    assert canonical_float32_reduce(values, identities) == recovar_logical_float32_reduce(canonical_values)
    assert canonical_float64_reduce(values, identities) == np.float64(3.0)


def test_promoted_float64_is_marked_and_guarded_from_genuine_source_claim():
    shifted, score_weight, projections, half_weight = _production_inputs()
    captured = normalized_cc_pixel_contributions(projections[0, 0], shifted[0, 0], score_weight[0], half_weight)
    promoted_report = replay_normalized_cc(captured)

    promoted_provenance = promoted_report.canonical_float64.provenance
    assert promoted_provenance.accumulation_dtype == "float64"
    assert promoted_provenance.precision_origin == "promoted_captured"
    assert not promoted_report.has_genuine_source_float64
    with pytest.raises(ValueError, match="promotes captured low-precision operands"):
        promoted_report.require_genuine_source_float64()

    recomputed = normalized_cc_pixel_contributions(
        projections[0, 0].astype(np.complex128),
        shifted[0, 0].astype(np.complex128),
        score_weight[0].astype(np.float64),
        half_weight.astype(np.float64),
        arithmetic_dtype=np.float64,
        precision_origin="recomputed_high_precision",
    )
    recomputed_report = replay_normalized_cc(recomputed)
    assert recomputed_report.has_genuine_source_float64
    recomputed_report.require_genuine_source_float64()

    with pytest.raises(ValueError, match="must use float32 arithmetic"):
        normalized_cc_pixel_contributions(
            projections[0, 0],
            shifted[0, 0],
            score_weight[0],
            half_weight,
            arithmetic_dtype=np.float64,
            precision_origin="captured_production",
        )
    with pytest.raises(ValueError, match="complex128/float64 source operands"):
        normalized_cc_pixel_contributions(
            projections[0, 0],
            shifted[0, 0],
            score_weight[0],
            half_weight,
            arithmetic_dtype=np.float64,
            precision_origin="recomputed_high_precision",
        )


def _candidate_contributions(numerator_values, *, high_precision=False):
    source_complex = np.complex128 if high_precision else np.complex64
    source_float = np.float64 if high_precision else np.float32
    arithmetic = np.float64 if high_precision else np.float32
    origin = "recomputed_high_precision" if high_precision else "captured_production"
    return normalized_cc_pixel_contributions(
        np.ones(len(numerator_values), dtype=source_complex),
        np.asarray(numerator_values, dtype=source_complex),
        np.ones(len(numerator_values), dtype=source_float),
        np.ones(len(numerator_values), dtype=source_float),
        arithmetic_dtype=arithmetic,
        precision_origin=origin,
    )


def test_candidate_replay_preserves_ties_and_classifies_precision_only_with_genuine_float64():
    # Captured operand sets disagree canonically, so promoted float64 alone is
    # correctly unresolved rather than being mislabeled as precision noise.
    rec_captured = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([1.0]), _candidate_contributions([1.0])],
        production_reducer="recovar_flat",
    )
    rel_captured = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([1.0 + 2e-7]), _candidate_contributions([1.0])],
        production_reducer="relion_coarse_128",
    )
    unresolved = classify_normalized_cc_candidate_replays(
        rec_captured,
        rel_captured,
        geometry_equal=True,
    )
    assert rec_captured.production_winners == (0, 1)
    assert rel_captured.production_winners == (0,)
    assert unresolved.classification == "operand_generation_or_upstream_precision_unresolved"

    # A genuine high-precision recomputation agrees on class 1 in both
    # engines, which upgrades the diagnosis to precision.
    rec_high = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([1.0], high_precision=True), _candidate_contributions([1.1], high_precision=True)],
        production_reducer="recovar_flat",
    )
    rel_high = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([1.0], high_precision=True), _candidate_contributions([1.1], high_precision=True)],
        production_reducer="relion_coarse_128",
    )
    classified = classify_normalized_cc_candidate_replays(
        rec_captured,
        rel_captured,
        geometry_equal=True,
        genuine_float64_recovar=rec_high,
        genuine_float64_relion=rel_high,
    )
    assert classified.classification == "precision"


def test_candidate_replay_classifies_geometry_before_scores():
    rec = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([1.0]), _candidate_contributions([2.0])],
        production_reducer="recovar_flat",
    )
    rel = replay_normalized_cc_candidates(
        [0, 1],
        [_candidate_contributions([2.0]), _candidate_contributions([1.0])],
        production_reducer="relion_coarse_128",
    )
    report = classify_normalized_cc_candidate_replays(rec, rel, geometry_equal=False)
    assert report.classification == "geometry"


def test_relion_cc_inverse_power_uses_selected_per_image_fourier_array():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    processed = np.asarray(
        [[1.0 + 2.0j, 3.0 - 4.0j, 5.0 + 6.0j]],
        dtype=np.complex64,
    )
    selected = np.asarray([2, 0], dtype=np.int32)
    observed = np.asarray(
        _relion_cc_inverse_power_from_processed(
            jnp.asarray(processed),
            selected,
        )
    )
    expected_power = sum(
        float(processed[0, index].real) ** 2
        + float(processed[0, index].imag) ** 2
        for index in selected
    )

    assert observed.dtype == np.float64
    np.testing.assert_array_equal(observed, np.asarray([[1.0 / expected_power]]))
